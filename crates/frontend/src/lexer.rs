use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use sirius::Sirius;
use std::ops::Range;

use logos::Logos;

use crate::parser::BinOp;

#[derive(Debug, Sirius, Clone, Copy, PartialEq, Encode, Decode)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}
impl Span {
    pub fn as_range(self: &Span) -> Range<usize> {
        return self.start..self.end;
    }
}

impl From<Range<usize>> for Span {
    fn from(value: Range<usize>) -> Self {
        Span {
            start: value.start,
            end: value.end,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Logos)]
#[logos(skip r"[ \t\r\f]+")]
#[logos(skip r"#.*")]
pub enum LogosToken {
    #[regex(r#"\d+"#, priority = 2)]
    Int,
    #[regex(r#"(\d*\.\d+([Ee](\+|-)?\d+)?)|(\d+[Ee](\+|-)?\d+)"#)]
    Float,
    #[regex(r#"[\p{L}a-zA-Z_][\p{L}\p{N}a-zA-Z0-9_]*"#)]
    Ident,
    #[regex(r#""([^"]|\\[\s\S])*""#)]
    String,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("...")]
    Ellipsis,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Times,
    #[token("/")]
    Slash,
    #[token("or")]
    Or,
    #[token("and")]
    And,
    #[token("&")]
    BitAnd,
    #[token("|")]
    BitOr,
    #[token("^")]
    BitXor,
    #[token(":")]
    Colon,
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token("->")]
    Arrow,
    #[token("fn")]
    KwFn,
    #[token("return")]
    KwReturn,
    #[token("not")]
    Not,
    #[token("extern")]
    KwExtern,
    #[token("let")]
    KwLet,
    #[token("\n")]
    Newline,
    #[token("==")]
    Eqq,
    #[token("!=")]
    Neq,
    #[token("=")]
    Eq,
    #[token(">")]
    Gt,
    #[token("<")]
    Lt,
    #[token(">=")]
    Ge,
    #[token("<=")]
    Le,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    Eof,
    Error,
}

impl std::fmt::Debug for LogosToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int => write!(f, "integer"),
            Self::Float => write!(f, "float"),
            Self::Ident => write!(f, "identifier"),
            Self::String => write!(f, "string"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::Ellipsis => write!(f, "..."),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Times => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Or => write!(f, "or"),
            Self::And => write!(f, "and"),
            Self::BitAnd => write!(f, "&"),
            Self::BitOr => write!(f, "|"),
            Self::BitXor => write!(f, "^"),
            Self::Colon => write!(f, ":"),
            Self::Dot => write!(f, "."),
            Self::Comma => write!(f, ","),
            Self::Semicolon => write!(f, ";"),
            Self::Arrow => write!(f, "->"),
            Self::KwFn => write!(f, "fn"),
            Self::KwReturn => write!(f, "return"),
            Self::Not => write!(f, "not"),
            Self::KwExtern => write!(f, "extern"),
            Self::KwLet => write!(f, "let"),
            Self::Newline => write!(f, "<newline>"),
            Self::Eqq => write!(f, "=="),
            Self::Neq => write!(f, "!="),
            Self::Eq => write!(f, "="),
            Self::Gt => write!(f, ">"),
            Self::Lt => write!(f, "<"),
            Self::Ge => write!(f, ">="),
            Self::Le => write!(f, "<="),
            Self::LBrace => write!(f, "{{"),
            Self::RBrace => write!(f, "}}"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::Eof => write!(f, "<eof>"),
            Self::Error => write!(f, "illegal token"),
        }
    }
}

impl LogosToken {
    pub fn to_infix_op(&self) -> Option<BinOp> {
        match self {
            Self::Plus => Some(BinOp::Add),
            Self::Minus => Some(BinOp::Sub),
            Self::Times => Some(BinOp::Mul),
            Self::Slash => Some(BinOp::Div),
            Self::Eqq => Some(BinOp::Eq),
            Self::Neq => Some(BinOp::Neq),
            Self::Lt => Some(BinOp::Lt),
            Self::Gt => Some(BinOp::Gt),
            Self::Le => Some(BinOp::Le),
            Self::Ge => Some(BinOp::Ge),
            Self::And => Some(BinOp::And),
            Self::Or => Some(BinOp::Or),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Token {
    pub kind: LogosToken,
    pub span: Span,
}

impl Token {
    pub fn is_eof(&self) -> bool {
        self.kind == LogosToken::Eof
    }

    pub fn len(&self) -> usize {
        self.span.end - self.span.start
    }

    pub fn text<'input>(&self, input: &'input str) -> &'input str {
        &input[self.span.start..self.span.end]
    }
}

pub struct LLexer<'a> {
    generated: logos::SpannedIter<'a, LogosToken>,
    peeked: Option<Token>,
    eof: bool,
    input_len: usize,
}

impl<'a> Iterator for LLexer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.peeked.take() {
            Some(token)
        } else {
            self.advance_token()
        }
    }
}

impl<'a> LLexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            generated: LogosToken::lexer(input).spanned(),
            eof: false,
            peeked: None,
            input_len: input.len(),
        }
    }

    fn advance_token(&mut self) -> Option<Token> {
        match self.generated.next() {
            Some((token, span)) => {
                let token = token.unwrap_or(LogosToken::Error);
                Some(Token {
                    kind: token,
                    span: span.into(),
                })
            }
            None if self.eof => None,
            None => {
                self.eof = true;
                Some(Token {
                    kind: LogosToken::Eof,
                    span: (self.input_len..self.input_len).into(),
                })
            }
        }
    }

    pub fn span(&self) -> Span {
        self.generated.span().into()
    }

    /// Peek at the next token without consuming it
    pub fn peek(&mut self) -> Option<Token> {
        if self.peeked.is_none() {
            self.peeked = self.advance_token();
        }
        self.peeked
    }

    /// Peek at the next token's kind without consuming it
    pub fn peek_kind(&mut self) -> Option<LogosToken> {
        self.peek().map(|token| token.kind)
    }

    /// Check if the next token matches a specific kind
    pub fn peek_is(&mut self, kind: LogosToken) -> bool {
        self.peek_kind() == Some(kind)
    }

    /// Check if the next token is EOF
    pub fn peek_is_eof(&mut self) -> bool {
        self.peek_is(LogosToken::Eof)
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        self.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        let mut tokens = LogosToken::lexer("1000 10.5");
        assert_eq!(tokens.next(), Some(Ok(LogosToken::Int)));
        assert_eq!(tokens.next(), Some(Ok(LogosToken::Float)));
    }

    #[test]
    fn float_parsing() {
        let mut tokens = LogosToken::lexer("3.14 .5 1e10 2.5e-3");
        assert_eq!(tokens.next(), Some(Ok(LogosToken::Float))); // 3.14
        assert_eq!(tokens.next(), Some(Ok(LogosToken::Float))); // .5
        assert_eq!(tokens.next(), Some(Ok(LogosToken::Float))); // 1e10
        assert_eq!(tokens.next(), Some(Ok(LogosToken::Float))); // 2.5e-3
    }

    #[test]
    fn llexer_basic_tokenization() {
        let mut lexer = LLexer::new("42 hello");

        let token1 = lexer.next().unwrap();
        assert_eq!(token1.kind, LogosToken::Int);
        assert_eq!(token1.span, Span::from(0..2));
        assert_eq!(token1.text("42 hello"), "42");
        assert!(!token1.is_eof());

        let token2 = lexer.next().unwrap();
        assert_eq!(token2.kind, LogosToken::Ident);
        assert_eq!(token2.span, Span::from(3..8));
        assert_eq!(token2.text("42 hello"), "hello");

        let eof_token = lexer.next().unwrap();
        assert_eq!(eof_token.kind, LogosToken::Eof);
        assert!(eof_token.is_eof());

        assert!(lexer.next().is_none());
    }

    #[test]
    fn llexer_tokenize_all() {
        let mut lexer = LLexer::new("fn test() -> int { return 42; }");
        let tokens = lexer.tokenize();

        let expected_kinds = vec![
            LogosToken::KwFn,
            LogosToken::Ident,
            LogosToken::LParen,
            LogosToken::RParen,
            LogosToken::Arrow,
            LogosToken::Ident, // int
            LogosToken::LBrace,
            LogosToken::KwReturn,
            LogosToken::Int,
            LogosToken::Semicolon,
            LogosToken::RBrace,
            LogosToken::Eof,
        ];

        assert_eq!(tokens.len(), expected_kinds.len());
        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(token.kind, *expected_kind);
        }
    }

    #[test]
    fn llexer_empty_input() {
        let mut lexer = LLexer::new("");

        let eof_token = lexer.next().unwrap();
        assert_eq!(eof_token.kind, LogosToken::Eof);
        assert_eq!(eof_token.span, Span::from(0..0));
        assert!(eof_token.is_eof());

        assert!(lexer.next().is_none());
    }

    #[test]
    fn llexer_with_comments() {
        let mut lexer = LLexer::new("42 # this is a comment\n+ 3.14");
        let tokens = lexer.tokenize();

        let expected_kinds = vec![
            LogosToken::Int,     // 42
            LogosToken::Newline, // \n
            LogosToken::Plus,    // +
            LogosToken::Float,   // 3.14
            LogosToken::Eof,
        ];

        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(token.kind, *expected_kind);
        }
    }

    #[test]
    fn llexer_string_literals() {
        let mut lexer = LLexer::new(r#""hello world" "escaped \"quote\"" "newline\n""#);

        let token1 = lexer.next().unwrap();
        assert_eq!(token1.kind, LogosToken::String);
        assert_eq!(
            token1.text(r#""hello world" "escaped \"quote\"" "newline\n""#),
            r#""hello world""#
        );

        let token2 = lexer.next().unwrap();
        assert_eq!(token2.kind, LogosToken::String);
        assert_eq!(
            token2.text(r#""hello world" "escaped \"quote\"" "newline\n""#),
            r#""escaped \"quote\""#
        );

        let token3 = lexer.next().unwrap();
        assert_eq!(token3.kind, LogosToken::String);
        assert_eq!(
            token3.text(r#""hello world" "escaped \"quote\"" "newline\n""#),
            r#""newline\n""#
        );
    }

    #[test]
    fn llexer_comparison_operators() {
        let mut lexer = LLexer::new("== != >= <= > <");
        let tokens = lexer.tokenize();

        let expected_kinds = vec![
            LogosToken::Eqq,
            LogosToken::Neq,
            LogosToken::Ge,
            LogosToken::Le,
            LogosToken::Gt,
            LogosToken::Lt,
            LogosToken::Eof,
        ];

        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(token.kind, *expected_kind);
        }
    }

    #[test]
    fn llexer_token_len() {
        let mut lexer = LLexer::new("hello == ...");

        let token1 = lexer.next().unwrap();
        assert_eq!(token1.len(), 5); // "hello"

        let token2 = lexer.next().unwrap();
        assert_eq!(token2.len(), 2); // "=="

        let token3 = lexer.next().unwrap();
        assert_eq!(token3.len(), 3); // "..."
    }

    #[test]
    fn llexer_error_handling() {
        let mut lexer = LLexer::new("valid @ invalid");

        let token1 = lexer.next().unwrap();
        assert_eq!(token1.kind, LogosToken::Ident); // "valid"

        let token2 = lexer.next().unwrap();
        assert_eq!(token2.kind, LogosToken::Error); // "@"

        let token3 = lexer.next().unwrap();
        assert_eq!(token3.kind, LogosToken::Ident); // "invalid"
    }

    // New tests for peeking functionality
    #[test]
    fn llexer_peek_basic() {
        let mut lexer = LLexer::new("42 hello");

        // Peek at first token
        let peeked = lexer.peek().unwrap();
        assert_eq!(peeked.kind, LogosToken::Int);
        assert_eq!(peeked.span, Span::from(0..2));

        // Peek again should return the same token
        let peeked_again = lexer.peek().unwrap();
        assert_eq!(peeked_again.kind, LogosToken::Int);
        assert_eq!(peeked_again.span, Span::from(0..2));

        // Consuming should return the peeked token
        let consumed = lexer.next().unwrap();
        assert_eq!(consumed.kind, LogosToken::Int);
        assert_eq!(consumed.span, Span::from(0..2));

        // Peek at second token
        let peeked2 = lexer.peek().unwrap();
        assert_eq!(peeked2.kind, LogosToken::Ident);
        assert_eq!(peeked2.span, Span::from(3..8));

        // Consume second token
        let consumed2 = lexer.next().unwrap();
        assert_eq!(consumed2.kind, LogosToken::Ident);
        assert_eq!(consumed2.span, Span::from(3..8));
    }

    #[test]
    fn llexer_peek_kind() {
        let mut lexer = LLexer::new("42 + hello");

        assert_eq!(lexer.peek_kind(), Some(LogosToken::Int));
        lexer.next(); // consume 42

        assert_eq!(lexer.peek_kind(), Some(LogosToken::Plus));
        lexer.next(); // consume +

        assert_eq!(lexer.peek_kind(), Some(LogosToken::Ident));
        lexer.next(); // consume hello

        assert_eq!(lexer.peek_kind(), Some(LogosToken::Eof));
    }

    #[test]
    fn llexer_peek_is() {
        let mut lexer = LLexer::new("fn test");

        assert!(lexer.peek_is(LogosToken::KwFn));
        assert!(!lexer.peek_is(LogosToken::Ident));

        lexer.next(); // consume fn

        assert!(lexer.peek_is(LogosToken::Ident));
        assert!(!lexer.peek_is(LogosToken::KwFn));
    }

    #[test]
    fn llexer_peek_is_eof() {
        let mut lexer = LLexer::new("42");

        assert!(!lexer.peek_is_eof());
        lexer.next(); // consume 42

        assert!(lexer.peek_is_eof());
    }

    #[test]
    fn llexer_peek_empty_input() {
        let mut lexer = LLexer::new("");

        assert!(lexer.peek_is_eof());
        let peeked = lexer.peek().unwrap();
        assert_eq!(peeked.kind, LogosToken::Eof);
    }

    #[test]
    fn llexer_mixed_peek_and_consume() {
        let mut lexer = LLexer::new("a + b - c");

        // Peek and consume interleaved
        assert!(lexer.peek_is(LogosToken::Ident)); // a
        let a = lexer.next().unwrap();
        assert_eq!(a.kind, LogosToken::Ident);

        assert!(lexer.peek_is(LogosToken::Plus)); // +
        assert!(lexer.peek_is(LogosToken::Plus)); // peek again
        let plus = lexer.next().unwrap();
        assert_eq!(plus.kind, LogosToken::Plus);

        let b = lexer.next().unwrap(); // consume without peeking
        assert_eq!(b.kind, LogosToken::Ident);

        assert!(lexer.peek_is(LogosToken::Minus)); // -
        let minus = lexer.next().unwrap();
        assert_eq!(minus.kind, LogosToken::Minus);

        assert!(lexer.peek_is(LogosToken::Ident)); // c
        let c = lexer.next().unwrap();
        assert_eq!(c.kind, LogosToken::Ident);

        assert!(lexer.peek_is_eof());
    }
}
