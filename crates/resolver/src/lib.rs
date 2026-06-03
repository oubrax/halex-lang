use std::collections::HashMap;
use std::ops::Range;

use ariadne::{Color, ColorGenerator, Fmt, Label, Report};
use frontend::{NodeId, ParseError, ReportedError, Span, Stmt, StmtKind, Type};

#[derive(Debug, Clone, PartialEq)]
pub enum ResolverError {
    ModuleNotFound { path: ModulePath, span: Span },
}

impl ReportedError for ResolverError {
    fn build_report<'a>(&self, file: &'a str) -> Report<'a, (&'a str, Range<usize>)> {
        let mut colors = ColorGenerator::new();
        let out = Color::Fixed(81);

        match self {
            Self::ModuleNotFound { path, span } => {
                Report::build(ariadne::ReportKind::Error, (file, span.as_range()))
                    .with_code("R0")
                    .with_message("module not found")
                    .with_label(
                        Label::new((file, span.as_range()))
                            .with_color(colors.next())
                            .with_message(format!(
                                "module `{}` not found",
                                path.as_string().fg(out)
                            )),
                    )
                    .finish()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModulePath(pub Vec<String>);

impl ModulePath {
    pub fn join(&self, seg: &str) -> Self {
        let mut v = self.0.clone();
        v.push(seg.to_string());
        ModulePath(v)
    }
    pub fn extend(&self, other: ModulePath) -> Self {
        let mut v = self.0.clone();
        v.extend(other.0);
        ModulePath(v)
    }
    pub fn as_string(&self) -> String {
        if self.0.is_empty() {
            "main".to_string()
        } else {
            self.0.join(".")
        }
    }
}

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: String,
    pub module: ModulePath,
    pub stmts: Vec<Stmt>,
}

impl SourceFile {
    pub fn new(path: String, module: ModulePath, stmts: Vec<Stmt>) -> Self {
        Self {
            path,
            module,
            stmts,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    Fn(Type),
    Module(HashMap<String, Symbol>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub node_id: NodeId,
    pub is_public: bool,
    pub kind: SymbolKind,
    pub decl_at: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    node_id: NodeId,
    absolute_path: ModulePath,
    span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleId(usize);

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModule {
    id: ModuleId,
    path: ModulePath,
    file_path: String,
    imports: Vec<Import>,

    symbols: HashMap<String, Symbol>,
}

pub struct Resolver {
    paths: HashMap<ModulePath, ModuleId>,
    modules: Vec<ResolvedModule>,
    pub errors: Vec<ResolverError>,
    pub parse_errors: HashMap<String, Vec<ParseError>>,
}

impl Resolver {
    pub fn new() -> Self {
        Self {
            paths: HashMap::new(),
            modules: Vec::new(),
            errors: Vec::new(),
            parse_errors: HashMap::new(),
        }
    }

    pub fn resolve(&mut self, initial_file_paths: Vec<String>) {
        let mut worklist: Vec<(ModulePath, Span, String)> = initial_file_paths
            .into_iter()
            .map(|path| {
                let module_name = std::path::Path::new(&path)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                (
                    ModulePath(vec![module_name]),
                    Span { start: 0, end: 0 },
                    path,
                )
            })
            .collect();

        let mut processed_modules = std::collections::HashSet::new();

        while let Some((module_path, import_span, file_path)) = worklist.pop() {
            if processed_modules.contains(&module_path) {
                continue;
            }

            let content = match std::fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(_) => {
                    if module_path.0.len() > 0 {
                        self.errors.push(ResolverError::ModuleNotFound {
                            path: module_path,
                            span: import_span,
                        });
                    }
                    continue;
                }
            };

            let parser = frontend::Parser::new(&content);
            let (stmts, parse_errors) = parser.parse_program();

            if !parse_errors.is_empty() {
                self.parse_errors.insert(file_path.clone(), parse_errors);
            }

            let source_file = SourceFile::new(file_path.clone(), module_path.clone(), stmts);

            self.resolve_file(&source_file);
            processed_modules.insert(module_path.clone());

            let module_id = self.paths.get(&source_file.module).unwrap();
            let module = &self.modules[module_id.0];

            for import in &module.imports {
                if !processed_modules.contains(&import.absolute_path) {
                    let mut path = std::path::PathBuf::new();
                    for part in &import.absolute_path.0 {
                        path.push(part);
                    }
                    path.set_extension("lx");
                    worklist.push((
                        import.absolute_path.clone(),
                        import.span,
                        path.to_str().unwrap().to_string(),
                    ));
                }
            }
        }

        for i in 0..self.modules.len() {
            self.resolve_imports_symbols(ModuleId(i));
        }
    }

    fn resolve_file(&mut self, file: &SourceFile) {
        let mut module = ResolvedModule {
            id: ModuleId(self.modules.len()),
            path: file.module.clone(),
            file_path: file.path.clone(),
            imports: Vec::new(),
            symbols: HashMap::new(),
        };
        for stmt in &file.stmts {
            self.resolve_declarations(stmt, &mut module);
        }
        self.modules.push(module.clone());
        self.paths.insert(module.path.clone(), module.id);
    }

    fn get_public_symbols(&self, module: &ResolvedModule) -> HashMap<String, Symbol> {
        let mut public_symbols = HashMap::new();
        for (k, v) in &module.symbols {
            if v.is_public {
                public_symbols.insert(k.to_owned(), v.to_owned());
            }
        }
        public_symbols
    }

    fn resolve_imports_symbols(&mut self, module_id: ModuleId) {
        let imports = self.modules[module_id.0].imports.clone();
        let mut new_symbols = HashMap::new();

        for import in &imports {
            if let Some(id) = self.paths.get(&import.absolute_path) {
                if let Some(other_module) = self.modules.get(id.0) {
                    new_symbols.insert(
                        import.absolute_path.0.last().unwrap().clone(),
                        Symbol {
                            node_id: import.node_id,
                            is_public: false,
                            decl_at: import.span,
                            kind: SymbolKind::Module(self.get_public_symbols(other_module)),
                        },
                    );
                }
            }
        }
        self.modules[module_id.0].symbols.extend(new_symbols);
    }
    fn resolve_declarations(&mut self, stmt: &Stmt, module: &mut ResolvedModule) {
        match &stmt.kind {
            StmtKind::Use { path } => {
                let module_path = ModulePath(path.inner.clone());

                module.imports.push(Import {
                    absolute_path: module_path,
                    span: stmt.span,
                    node_id: stmt.id,
                });
            }
            StmtKind::Function {
                is_public,
                name,
                params,
                return_type,
                ..
            } => {
                module.symbols.insert(
                    name.inner.clone(),
                    Symbol {
                        node_id: stmt.id,
                        decl_at: stmt.span,
                        kind: SymbolKind::Fn(Type::Function {
                            params: params.iter().map(|(_, t)| t.clone()).collect(),
                            return_type: Box::new(return_type.clone()),
                        }),
                        is_public: *is_public,
                    },
                );
            }
            _ => {}
        }
    }
}
