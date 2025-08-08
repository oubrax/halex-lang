use std::{
    env,
    fs::{self, File},
    hash::{DefaultHasher, Hash, Hasher},
    io::{Read, Write},
    path::Path,
    time::Instant,
};

use ariadne::{Color, Fmt, Source};
use bincode::{Decode, Encode};
use frontend::{Expr, ParseError, Parser, ReportedError};
use tycheck::Checker;

#[derive(Encode, Decode)]
struct CachedAst {
    hash: u64,
    ast: Vec<frontend::Stmt>,
}

fn get_cache_path(file_path: &str) -> String {
    let safe_name = file_path.replace('/', "_");
    format!(".halex/ast_cache_{}", safe_name)
}

fn write_ast_cache(
    file_path: &str,
    hash: u64,
    ast: &[frontend::Stmt],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(".halex")?;
    let mut file = File::create(get_cache_path(file_path))?;

    let cached = CachedAst {
        hash,
        ast: ast.to_vec(),
    };

    bincode::encode_into_std_write(cached, &mut file, bincode::config::standard())?;
    Ok(())
}

fn read_ast_cache(
    file_path: &str,
    expected_hash: u64,
) -> Result<Option<Vec<frontend::Stmt>>, Box<dyn std::error::Error>> {
    let mut buf = Vec::new();
    File::open(get_cache_path(file_path))?.read_to_end(&mut buf)?;

    match bincode::decode_from_slice(&buf, bincode::config::standard()) {
        Ok((CachedAst { hash, ast }, _)) if hash == expected_hash => Ok(Some(ast)),
        _ => Ok(None),
    }
}

fn file_hash(input: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

fn parse_file(file_path: &str) -> (String, Vec<frontend::Stmt>, Vec<ParseError>) {
    let input = fs::read_to_string(file_path).expect("Failed to read source file");
    let hash = file_hash(&input);

    if let Ok(Some(ast)) = read_ast_cache(file_path, hash) {
        println!("parse({file_path}) = CACHE HIT");
        return (input, ast, vec![]);
    }

    let parser = Parser::new(&input);
    let (ast, errors) = parser.parse_program();

    if errors.is_empty() {
        let _ = write_ast_cache(file_path, hash, &ast);
    }

    (input, ast, errors)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("Usage: <program> <filename>");
            std::process::exit(1);
        }
    };

    let parse_start = Instant::now();
    let (input, ast, parse_errors) = parse_file(&file_path);
    // println!("{:#?}", &ast);
    println!(
        "parsed in {}",
        format!("{:?}", parse_start.elapsed()).fg(Color::Yellow)
    );

    for error in &parse_errors {
        error
            .build_report(&file_path)
            .print((file_path.as_str(), Source::from(&input)))?;
    }

    if !parse_errors.is_empty() {
        std::process::exit(1);
    }

    let typecheck_start = Instant::now();
    let mut checker = Checker::new();

    match checker.solve(&ast) {
        Ok(map) => {}
        Err(errors) => {
            for e in &errors {
                e.build_report(&file_path)
                    .eprint((file_path.as_str(), Source::from(&input)))?;
                eprintln!();
            }
        }
    };

    println!(
        "typechecked in {}",
        format!("{:?}", typecheck_start.elapsed()).fg(Color::Yellow)
    );

    Ok(())
}
