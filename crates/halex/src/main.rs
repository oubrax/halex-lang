use ariadne::{Color, Fmt, Source};
use frontend::{Parser, ReportedError};
use miette::{IntoDiagnostic, Result};

use tycheck::Checker;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }

    let input = std::fs::read_to_string(&args[1]).into_diagnostic()?;
    let mut parser = Parser::new(&input);
    let start = std::time::Instant::now();
    let (result, errors) = parser.parse_program();
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
