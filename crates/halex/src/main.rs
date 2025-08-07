use std::{
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    io::{Read, Write},
};

use ariadne::{Color, Fmt, Source};
use bincode::{BorrowDecode, Decode, Encode, borrow_decode_from_slice, error::DecodeError};
use frontend::{Expr, Parser, ReportedError};
use tycheck::Checker;

#[derive(Encode, BorrowDecode)]
struct CachedAst<'a> {
    hash: u64,
    ast: Vec<Expr<'a>>,
}

// read cache, keeping bytes alive for AST borrows
fn read_cache<'a>(
    path: &str,
    expected_hash: u64,
) -> Result<Option<(Vec<Expr<'a>>)>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    // try borrow‐decode from slice
    match borrow_decode_from_slice(&buf[..], bincode::config::standard()) {
        Ok((CachedAst { hash, ast }, _)) if hash == expected_hash => Ok(Some(ast)),
        _ => Ok(None),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }

    let input = std::fs::read_to_string(&args[1])?;

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    println!("{:?}", hasher.finish());
    let parser = Parser::new(&input);
    let start = std::time::Instant::now();
    let (result, errors) = parser.parse_program();
    let mut file = File::create("ast")?;
    bincode::encode_into_std_write(result.clone(), &mut file, bincode::config::standard())?;
    println!("{:#?}", &result);
    let duration = start.elapsed();

    for e in &errors {
        e.build_report(&args[1])
            .print((args[1].as_str(), Source::from(&input)))
            .unwrap()
    }

    println!("parsed in {}", format!("{:?}", duration).fg(Color::Yellow));

    if !errors.is_empty() {
        std::process::exit(1)
    }

    let start = std::time::Instant::now();
    let mut checker = Checker::new();

    let duration = start.elapsed();
    match checker.solve(&result) {
        Ok(map) => println!("{:#?}", map),
        Err(errors) => {
            for e in &errors {
                e.build_report(&args[1])
                    .eprint((args[1].as_str(), Source::from(&input)))
                    .unwrap();
                eprintln!()
            }
        }
    };

    println!(
        "typechecked in {}",
        format!("{:?}", duration).fg(Color::Yellow)
    );

    Ok(())
}
