use std::ops::Range;

use ariadne::Report;

pub trait ReportedError {
    fn build_report<'a>(&'a self, file: &'a str) -> Report<'a, (&str, Range<usize>)>;
}
