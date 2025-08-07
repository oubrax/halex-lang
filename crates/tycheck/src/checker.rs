use std::{collections::HashMap, fmt::Display};

use ariadne::{Color, ColorGenerator, Fmt, Label, Report};
use frontend::{
    Expr, ExprKind, Literal, NodeId, ReportedError, Span, Spanned, Stmt, StmtKind, Type,
};

#[derive(Clone, Debug)]
pub enum CheckerError<'a> {
    TypeMismatch {
        found: Ty,
        expected: Ty,
        expected_span: Span,
        found_span: Span,
        is_found_var: bool,
        is_expected_var: bool,
        unify_ctx: Option<UnifyContext<'a>>,
    },

    CouldntInfer {
        var_span: Span,
    },

    NotFound {
        name: &'a str,
        ident_span: Span,
    },
}

impl<'a> ReportedError for CheckerError<'a> {
    fn build_report<'b>(&'b self, file: &'b str) -> Report<'b, (&'b str, std::ops::Range<usize>)> {
        let mut colors = ColorGenerator::new();
        let out = Color::Fixed(81);
        match self {
            Self::TypeMismatch {
                found,
                expected,
                expected_span,
                found_span,
                is_expected_var,
                is_found_var,
                unify_ctx,
            } => {
                let found_color = colors.next();
                let expected_color = colors.next();

                let (code, message) = match unify_ctx {
                    Some(ctx) => match ctx {
                        UnifyContext::Return => {
                            if matches!(found, Ty::Unit) {
                                ("T3", "'return;' in function whose return type is not ()")
                            } else {
                                ("T0", "mismatched types")
                            }
                        }
                        UnifyContext::Binary => ("T1", "mismatched types in binary operands"),
                        UnifyContext::Variable(_) => {
                            ("T2", "mismatched types in variable declaration")
                        }
                    },
                    _ => ("T0", "mismatched types"),
                };

                let mut r =
                    Report::build(ariadne::ReportKind::Error, (file, found_span.as_range()))
                        .with_code(code)
                        .with_message(message)
                        .with_label(
                            Label::new((file, expected_span.as_range()))
                                .with_message(format!(
                                    "{} type {}",
                                    {
                                        if *is_expected_var {
                                            "expected inferred"
                                        } else {
                                            "expected"
                                        }
                                    },
                                    expected.fg(out)
                                ))
                                .with_color(expected_color),
                        )
                        .with_label(
                            Label::new((file, found_span.as_range()))
                                .with_message(format!(
                                    "{} type {}",
                                    {
                                        if *is_found_var {
                                            "found inferred"
                                        } else {
                                            "found"
                                        }
                                    },
                                    found.fg(out)
                                ))
                                .with_color(found_color),
                        );

                if let Some(UnifyContext::Variable(name)) = unify_ctx {
                    r.with_helps([format!(
                        "correct type annotation: let {}: {} = ...",
                        name,
                        found.fg(found_color)
                    )]);
                };

                r.finish()
            }

            Self::CouldntInfer { var_span } => {
                let color = colors.next();

                Report::build(ariadne::ReportKind::Error, (file, var_span.as_range()))
                    .with_code("T4")
                    .with_message("Couldn't infer type")
                    .with_label(
                        Label::new((file, var_span.as_range()))
                            .with_message(format!("add type annotation here"))
                            .with_color(color),
                    )
                    .with_help("please add type annotation")
                    .finish()
            }
            Self::NotFound {
                name: _,
                ident_span,
            } => {
                let color = colors.next();

                Report::build(ariadne::ReportKind::Error, (file, ident_span.as_range()))
                    .with_code("T5")
                    .with_message("Couldn't find item in the current scope")
                    .with_label(
                        Label::new((file, ident_span.as_range()))
                            .with_message(format!("this item doesn't exist"))
                            .with_color(color),
                    )
                    .finish()
            }
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Copy, Clone)]
pub struct VarId(usize);

#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Copy, Clone)]
enum Origin {
    Integer,
    Float,
    Infer,
}

impl Default for Origin {
    fn default() -> Self {
        Origin::Infer
    }
}

#[derive(Hash, Eq, PartialEq, PartialOrd, Copy, Clone)]
pub struct Var {
    id: VarId,
    origin: Origin,
}

impl Var {
    /// New type variable with default origin (infer)
    pub fn new(id: VarId) -> Self {
        Self {
            id,
            origin: Origin::default(),
        }
    }

    /// New int type variable - only unifies with concrete int types
    pub fn int(id: VarId) -> Self {
        Self {
            id,
            origin: Origin::Integer,
        }
    }
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.origin {
            Origin::Integer => write!(f, "{{int}}"),
            Origin::Infer => write!(f, "'{}", self.id.0),
            Origin::Float => write!(f, "{{float}}"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Ty {
    I8,
    I16,
    I32,
    I64,

    Str,

    Fn {
        params: Vec<Spanned<Ty>>,
        return_ty: Box<Spanned<Ty>>,
    },
    // Bool,
    Unit,
    Never,
    Var(Var),
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I8 => write!(f, "i8"),
            Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::Str => write!(f, "str"),
            Self::Unit => write!(f, "()"),
            Self::Never => write!(f, "!"),
            Self::Var(v) => write!(f, "{v:?}"),
            Self::Fn { params, return_ty } => {
                write!(
                    f,
                    "fn ({}) -> {}",
                    params
                        .iter()
                        .map(|p| format!("{}", p.inner))
                        .collect::<Vec<_>>()
                        .join(", "),
                    &return_ty.inner
                )
            }
        }
    }
}
impl Ty {
    pub fn is_integer(&self) -> bool {
        match self {
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum UnifyContext<'a> {
    Variable(&'a str),
    Binary,
    Return,
}

#[derive(Clone, Debug)]
enum Constraint<'a> {
    Unify(Spanned<Ty>, Spanned<Ty>, Option<UnifyContext<'a>>),
    Int(Spanned<Ty>),
}

type Item = Spanned<Ty>;
#[derive(Debug)]
pub struct Scope<'a> {
    stack: Vec<HashMap<&'a str, Item>>,
}

impl<'a> Scope<'a> {
    pub fn new() -> Self {
        Self {
            stack: vec![HashMap::new()], // global scope
        }
    }

    pub fn enter_scope(&mut self) {
        self.stack.push(HashMap::new());
    }

    pub fn exit_scope(&mut self) {
        self.stack
            .pop()
            .expect("Tried to pop from empty scope stack");
    }

    pub fn insert(&mut self, name: &'a str, item: Item) {
        self.stack
            .last_mut()
            .expect("No current scope")
            .insert(name, item);
    }

    pub fn get(&self, name: &&'a str) -> Option<&Item> {
        for scope in self.stack.iter().rev() {
            if let Some(item) = scope.get(name) {
                return Some(item);
            }
        }
        None
    }

    pub fn get_mut(&mut self, name: &&'a str) -> Option<&mut Item> {
        for scope in self.stack.iter_mut().rev() {
            if let Some(item) = scope.get_mut(name) {
                return Some(item);
            }
        }
        None
    }

    pub fn contains(&self, name: &&'a str) -> bool {
        self.get(name).is_some()
    }

    pub fn current_scope_mut(&mut self) -> &mut HashMap<&'a str, Item> {
        self.stack.last_mut().expect("No current scope to mutate")
    }
}
pub struct Checker<'a> {
    scope: Scope<'a>,
    subst: HashMap<VarId, Ty>,
    unresolved_vars: HashMap<VarId, Span>,
    constraints: Vec<Constraint<'a>>,
    errors: Vec<CheckerError<'a>>,
    typemap: HashMap<NodeId, Ty>,
    next_var: usize,
    in_function: Option<Spanned<Ty>>,
}

impl<'a> Checker<'a> {
    pub fn new() -> Self {
        Self {
            scope: Scope::new(),
            subst: HashMap::new(),
            in_function: None,
            constraints: Vec::new(),
            errors: Vec::new(),
            next_var: 0,
            typemap: HashMap::new(),
            unresolved_vars: HashMap::new(),
        }
    }
    pub fn solve(
        &mut self,
        stmts: &[Stmt<'a>],
    ) -> Result<HashMap<NodeId, Ty>, Vec<CheckerError<'a>>> {
        for stmt in stmts {
            self.infer_stmt(stmt);
        }

        let mut constraints = self.solve_constraints();

        constraints.retain(|c| !self.finalize_constraint(c));

        self.constraints = constraints;

        let constraints = self.solve_constraints();
        println!("{:#?}", constraints);
        self.concreticize();

        self.handle_errors(constraints);

        if self.errors.is_empty() {
            Ok(std::mem::take(&mut self.typemap))
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn handle_errors(&mut self, constraint: Vec<Constraint<'a>>) {
        for c in constraint {
            match c {
                Constraint::Unify(a, b, unify_ctx) => {
                    let a_ty = self.find(&a.inner);
                    let b_ty = self.find(&b.inner);
                    if a_ty == b_ty {
                        continue;
                    }

                    self.errors.push(CheckerError::TypeMismatch {
                        is_found_var: matches!(&b.inner, Ty::Var(_)),
                        is_expected_var: matches!(&a.inner, Ty::Var(_)),
                        found: b_ty,
                        expected: a_ty,
                        found_span: b.span,
                        expected_span: a.span,
                        unify_ctx,
                    });
                }
                Constraint::Int(i) => {
                    let ty = self.find(&i.inner);
                    if !ty.is_integer() {
                        self.errors.push(CheckerError::TypeMismatch {
                            expected: Ty::I32, // or just `int` placeholder
                            found: ty,
                            expected_span: i.span,
                            found_span: i.span,
                            is_found_var: false,
                            is_expected_var: false,
                            unify_ctx: None,
                        });
                    }
                }
            }
        }
    }
    fn concreticize(&mut self) {
        println!("{:?}", self.scope);

        for (id, var_span) in &self.unresolved_vars {
            if matches!(self.find(&Ty::Var(Var::new(*id))), Ty::Var(_)) {
                self.errors.push(CheckerError::CouldntInfer {
                    var_span: *var_span,
                })
            }
        }
        for (i, d) in self.typemap.clone() {
            let resolved = self.find(&d);
            self.typemap.insert(i, resolved);
        }
    }

    fn new_var(&mut self, span: Span) -> VarId {
        let id = VarId(self.next_var);
        self.unresolved_vars.insert(id, span);
        self.next_var += 1;
        id
    }

    fn to_ty(&mut self, ty: &Spanned<Type>) -> Spanned<Ty> {
        Spanned::new(
            match ty.inner.clone() {
                Type::I8 => Ty::I8,
                Type::I16 => Ty::I16,
                Type::I32 => Ty::I32,
                Type::I64 => Ty::I64,
                Type::Str => Ty::Str,
                Type::Unit => Ty::Unit,
                Type::Infer => Ty::Var(Var::new(self.new_var(ty.span))),
                Type::Function {
                    params,
                    return_type,
                } => Ty::Fn {
                    params: params.iter().map(|p| self.to_ty(p)).collect::<Vec<_>>(),
                    return_ty: Box::new(self.to_ty(&return_type)),
                },
            },
            ty.span,
        )
    }

    fn infer_stmt(&mut self, stmt: &Stmt<'a>) -> Ty {
        let ty = match &stmt.kind {
            StmtKind::Expr(expr) => self.infer_expr(expr).inner,
            StmtKind::Function {
                name,
                params,
                return_type,
                body,
            } => {
                let name_str = name.inner;

                let mut params_ty = Vec::new();
                for (_, param_ty) in params {
                    let ty = self.to_ty(param_ty);
                    params_ty.push(ty);
                }
                let prev_in = self.in_function.clone();
                let return_ty = self.to_ty(return_type);
                self.in_function = Some(return_ty.clone());
                let span = Span::from(name.span.start..return_type.span.end);
                self.scope.insert(
                    name_str,
                    Spanned::new(
                        Ty::Fn {
                            params: params_ty.iter().map(|p| p.clone()).collect(),
                            return_ty: Box::new(return_ty.clone()),
                        },
                        span,
                    ),
                );

                self.scope.enter_scope();

                for ((param_name, _), param_ty) in params.iter().zip(params_ty.iter()) {
                    self.scope.insert(*param_name, param_ty.clone());
                }

                let block = self.infer_expr(body);

                self.scope.exit_scope();
                self.in_function = prev_in;
                self.constraints
                    .push(Constraint::Unify(return_ty, block, None));
                Ty::Unit
            }
            StmtKind::Let {
                name,
                type_ann,
                value,
            } => {
                let name_str = name.inner;

                let _ty_ann: Option<Spanned<Ty>> = type_ann.as_ref().map(|t| self.to_ty(t));
                dbg!(&_ty_ann);
                let ty_ann = _ty_ann.clone().unwrap_or_else(|| {
                    Spanned::new(Ty::Var(Var::new(self.new_var(name.span))), name.span)
                });

                let ty_expr = self.infer_expr(value);

                self.scope
                    .insert(name_str, _ty_ann.unwrap_or(ty_expr.clone()));

                self.constraints.push(Constraint::Unify(
                    ty_ann,
                    ty_expr,
                    Some(UnifyContext::Variable(name_str)),
                ));
                Ty::Unit
            }

            StmtKind::Return(e) => {
                match (self.in_function.clone(), e) {
                    (Some(return_type), Some(expr)) => {
                        let ty = self.infer_expr(expr);
                        self.constraints.push(Constraint::Unify(
                            return_type,
                            ty,
                            Some(UnifyContext::Return),
                        ));
                    }
                    (Some(t), None) => {
                        self.constraints.push(Constraint::Unify(
                            t,
                            Spanned::new(Ty::Unit, stmt.span),
                            Some(UnifyContext::Return),
                        ));
                    }
                    _ => {
                        todo!("make it error out that its not in func")
                    }
                }

                Ty::Never
            }

            _ => Ty::Unit,
        };

        self.typemap.insert(stmt.id, ty.clone());

        ty
    }

    fn infer_expr(&mut self, expr: &Expr<'a>) -> Spanned<Ty> {
        let (ty, span) = match &expr.kind {
            ExprKind::Identifier(name) => (
                match self.scope.get(name) {
                    None => {
                        self.errors.push(CheckerError::NotFound {
                            name: *name,
                            ident_span: expr.span,
                        });
                        Ty::Never
                    }

                    Some(item) => {
                        println!("{item:?}");
                        item.inner.clone()
                    }
                },
                expr.span,
            ),
            ExprKind::Literal(l) => (
                match l {
                    Literal::Int(_) => {
                        let v = Var::int(self.new_var(expr.span));

                        self.constraints
                            .push(Constraint::Int(Spanned::new(Ty::Var(v), expr.span.clone())));

                        Ty::Var(v)
                    }
                    Literal::Str(_) => Ty::Str,
                    _ => todo!(),
                },
                expr.span,
            ),

            ExprKind::Call { name, args } => (
                match self.scope.get(&name.inner).cloned() {
                    Some(ty) => match self.find(&ty.inner).clone() {
                        Ty::Fn { params, return_ty } => {
                            if params.len() != args.len() {
                                // TODO: proper error
                                Ty::Never
                            } else {
                                for (arg, param) in args.iter().zip(params.iter()) {
                                    let arg_ty = self.infer_expr(arg);

                                    self.constraints.push(Constraint::Unify(
                                        param.clone(),
                                        arg_ty,
                                        None,
                                    ));
                                }
                                return_ty.inner
                            }
                        }
                        Ty::Var(_var) => {
                            let arg_tys = args.iter().map(|arg| self.infer_expr(arg)).collect();
                            let ret_ty =
                                Spanned::new(Ty::Var(Var::new(self.new_var(expr.span))), expr.span);

                            let fn_ty = Ty::Fn {
                                params: arg_tys,
                                return_ty: Box::new(ret_ty.clone()),
                            };

                            self.constraints.push(Constraint::Unify(
                                ty,
                                Spanned::new(fn_ty, name.span),
                                None,
                            ));

                            ret_ty.inner
                        }
                        _ => {
                            println!("Oops");
                            // emit error
                            Ty::Never
                        }
                    },
                    None => {
                        self.errors.push(CheckerError::NotFound {
                            name: name.inner,
                            ident_span: name.span,
                        });
                        Ty::Never
                    }
                },
                expr.span,
            ),

            ExprKind::Block(stmts) => {
                self.scope.enter_scope();
                let res = stmts.iter().fold(None, |_, s| {
                    let ty = self.infer_stmt(s);
                    let span = if let StmtKind::Expr(e) = &s.kind {
                        e.span
                    } else {
                        s.span
                    };
                    Some((ty, span))
                });
                self.scope.exit_scope();

                res.unwrap_or((Ty::Unit, expr.span))
            }
            ExprKind::Binary { left, op, right } => (
                {
                    let lhs_ty = self.infer_expr(left.as_ref());
                    let rhs_ty = self.infer_expr(right.as_ref());

                    self.constraints.push(Constraint::Unify(
                        lhs_ty.clone(),
                        rhs_ty,
                        Some(UnifyContext::Binary),
                    ));

                    lhs_ty.inner
                },
                expr.span,
            ),

            ExprKind::Unit => (Ty::Unit, expr.span),
            _ => todo!(),
        };
        self.typemap.insert(expr.id, ty.clone());
        Spanned::new(ty, span)
    }

    fn occurs_check(&self, var_id: VarId, ty: &Ty) -> bool {
        match ty {
            Ty::Var(var) => {
                if var.id == var_id {
                    return true;
                }
                // Follow substitution chain
                if let Some(substituted) = self.subst.get(&var.id) {
                    self.occurs_check(var_id, substituted)
                } else {
                    false
                }
            }
            Ty::Fn { params, return_ty } => {
                // Check if var occurs in any parameter type
                for param in params {
                    if self.occurs_check(var_id, &param.inner) {
                        return true;
                    }
                }
                // Check if var occurs in return type
                self.occurs_check(var_id, &return_ty.inner)
            }
            // For other types, var cannot occur
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::Str | Ty::Unit | Ty::Never => false,
        }
    }

    fn find(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Var(idx) => {
                if let Some(v) = self.subst.get(&idx.id) {
                    self.find(v)
                } else {
                    ty.clone()
                }
            }
            Ty::Fn { params, return_ty } => Ty::Fn {
                params: params
                    .iter()
                    .map(|p| Spanned::new(self.find(&p.inner), p.span))
                    .collect::<Vec<Spanned<Ty>>>(),
                return_ty: Box::new(Spanned::new(self.find(&return_ty.inner), return_ty.span)),
            },
            _ => ty.clone(),
        }
    }

    fn finalize_constraint(&mut self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::Int(ty) => {
                let ty = self.find(&ty.inner);

                match ty {
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => true,
                    Ty::Var(Var {
                        id,
                        origin: Origin::Integer,
                    }) => {
                        self.subst.insert(id, Ty::I32);
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
    fn solve_constraints(&mut self) -> Vec<Constraint<'a>> {
        let constraints = loop {
            let mut constraints = std::mem::take(&mut self.constraints);
            let initial_length = constraints.len();
            println!("{:?}", constraints);
            constraints.retain(|constraint| !self.solve_constraint(&constraint));

            if constraints.len() == initial_length {
                break constraints;
            }

            self.constraints.extend(constraints);
        };

        dbg!(&constraints);
        constraints
    }

    fn is_compatible(&self, origin: Origin, ty: &Ty) -> bool {
        match origin {
            Origin::Integer => ty.is_integer(),
            Origin::Float => todo!(),
            Origin::Infer => true,
        }
    }
    // fn occurs_check(&mut self, var: Ty, )
    fn solve_constraint(&mut self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::Unify(a, b, _) => {
                let a = self.find(&a.inner);
                let b = self.find(&b.inner);
                if a == b {
                    return true;
                }
                println!("UNIFY {:?} = {:?}", a, b);

                match (a, b) {
                    (Ty::Never, _) | (_, Ty::Never) => true,
                    (
                        Ty::Fn {
                            params: params_a,
                            return_ty: return_a,
                        },
                        Ty::Fn {
                            params: params_b,
                            return_ty: return_b,
                        },
                    ) => {
                        if params_a.len() != params_b.len() {
                            return false;
                        }

                        for (p_a, p_b) in params_a.iter().zip(params_b.iter()) {
                            self.constraints.push(Constraint::Unify(
                                p_a.clone(),
                                p_b.clone(),
                                None,
                            ));
                        }

                        self.constraints
                            .push(Constraint::Unify(*return_a, *return_b, None));

                        true
                    }
                    (Ty::Var(a), Ty::Var(b)) => {
                        let origin = match (a.origin, b.origin) {
                            (a, b) if a == b => b,
                            _ => return false,
                        };
                        println!("{:?}={:?}", a, b);
                        self.subst.insert(a.id, Ty::Var(Var { origin, id: b.id }));
                        true
                    }
                    (Ty::Var(var), concrete) | (concrete, Ty::Var(var)) => {
                        if self.occurs_check(var.id, &concrete) {
                            return false; // This will cause a type error
                        }
                        match var.origin {
                            Origin::Integer => {
                                if concrete.is_integer() {
                                    self.subst.insert(var.id, concrete);
                                    self.unresolved_vars.remove(&var.id);
                                    true
                                } else {
                                    false
                                }
                            }
                            Origin::Infer => {
                                self.subst.insert(var.id, concrete);
                                self.unresolved_vars.remove(&var.id);
                                true
                            }
                            Origin::Float => todo!(),
                        }
                    }
                    _ => false,
                }
            }

            Constraint::Int(ty) => false,
        }
    }
}
