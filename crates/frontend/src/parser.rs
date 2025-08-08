use std::{ops::Range, vec};

use crate::{
    ReportedError, Span,
    lexer::{LLexer, LogosToken, Token},
};
use ariadne::{Color, ColorGenerator, Fmt, Label, Report};
use bincode::{Decode, Encode};

type Param = (String, Spanned<Type>);

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Clone)]
pub enum ParseError {
    UnexpectedToken {
        span: Span,
        found: String,
        expected: Vec<String>,
    },
    MissingDelimiter {
        span: Span,
        delimiter: String,
        opening_span: Span,
    },

    InvalidNumber {
        span: Span,
        value: String,
    },

    MissingSeparator {
        span: Span,
    },

    UnexpectedEof {
        span: Span,
        expected: String,
    },
}

impl ReportedError for ParseError {
    fn build_report<'a>(&self, file: &'a str) -> Report<'a, (&'a str, Range<usize>)> {
        let mut colors = ColorGenerator::new();

        let out = Color::Fixed(81);

        match self {
            Self::UnexpectedToken {
                span,
                found: _,
                expected,
            } => {
                let color = colors.next();
                Report::build(ariadne::ReportKind::Error, (file, span.as_range()))
                    .with_code("P0")
                    .with_message("unexpected token")
                    .with_label(
                        Label::new((file, span.as_range()))
                            .with_color(color)
                            .with_message(format!("expected {}", expected.join(", ").fg(out))),
                    )
                    .finish()
            }

            Self::MissingDelimiter {
                span,
                delimiter,
                opening_span,
            } => {
                let a = colors.next();
                let b = colors.next();
                Report::build(ariadne::ReportKind::Error, (file, span.as_range()))
                    .with_code("P1")
                    .with_message(format!("missing closing delimiter {}", delimiter.fg(out)))
                    .with_label(
                        Label::new((file, opening_span.as_range()))
                            .with_color(a)
                            .with_message("opened here"),
                    )
                    .with_label(
                        Label::new((file, span.as_range()))
                            .with_color(b)
                            .with_message(format!("expected closing {}", delimiter.fg(out))),
                    )
                    .finish()
            }

            Self::InvalidNumber { span, value } => {
                let color = colors.next();
                Report::build(ariadne::ReportKind::Error, (file, span.as_range()))
                    .with_code("P2")
                    .with_message("invalid number")
                    .with_label(
                        Label::new((file, span.as_range()))
                            .with_color(color)
                            .with_message(format!("{} is not a valid number", value.fg(out))),
                    )
                    .finish()
            }

            Self::MissingSeparator { span } => {
                let color = colors.next();
                Report::build(ariadne::ReportKind::Error, (file, span.as_range()))
                    .with_code("P3")
                    .with_message("missing separator")
                    .with_label(
                        Label::new((file, span.as_range()))
                            .with_color(color)
                            .with_message("expected separator here"),
                    )
                    .finish()
            }

            Self::UnexpectedEof { span, expected } => {
                let color = colors.next();
                Report::build(ariadne::ReportKind::Error, (file, span.as_range()))
                    .with_code("P4")
                    .with_message("unexpected end of file")
                    .with_label(
                        Label::new((file, span.as_range()))
                            .with_color(color)
                            .with_message(format!("expected {}", expected).fg(out)),
                    )
                    .finish()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct Spanned<T> {
    pub inner: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(inner: T, span: Span) -> Self {
        Self { inner, span }
    }
}
#[derive(Debug, Clone, Encode, Decode, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Str(String),
}

/// Binary operator used for infix operations (1 + 1) etc
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

/// Binary operator used for infix operations (1 + 1) etc
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Debug, Hash, Clone, Copy, Eq, Encode, Decode, PartialEq)]
pub struct NodeId(usize);

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct Expr {
    pub id: NodeId,
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, Encode, Decode, PartialEq)]
pub enum ExprKind {
    Literal(Literal),
    Identifier(String),
    Binary {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    Call {
        name: Spanned<String>,
        args: Vec<Expr>,
    },
    Unit,
    Block(Vec<Stmt>),
    Error,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct Stmt {
    pub id: NodeId,
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, Encode, Decode, PartialEq)]
pub enum StmtKind {
    Let {
        name: Spanned<String>,
        type_ann: Option<Spanned<Type>>,
        value: Expr,
    },
    Assign {
        name: Expr,
        value: Expr,
    },
    Return(Option<Expr>),
    Function {
        name: Spanned<String>,
        params: Vec<Param>,
        return_type: Spanned<Type>,
        body: Expr,
    },
    Extern {
        name: Spanned<String>,
        params: Vec<Param>,
        return_type: Spanned<Type>,
    },
    Expr(Expr),
    Error,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    Str, // immutable global string aka [x]u8
    Unit,
    Infer,
    Function {
        params: Vec<Spanned<Type>>,
        return_type: Box<Spanned<Type>>,
    },
}

pub struct Parser<'a> {
    lexer: LLexer<'a>,
    errors: Vec<ParseError>,
    current: Option<Token>,
    input: &'a str,
    /// Track delimiter stack for better error recovery
    delimiter_stack: Vec<(LogosToken, Span)>,
    /// Panic mode - when true, we're recovering from an error
    panic_mode: bool,
    next_id: usize,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            errors: Vec::new(),
            lexer: LLexer::new(input),
            current: None,
            input,
            next_id: 0,
            delimiter_stack: Vec::new(),
            panic_mode: false,
        }
    }

    fn get_node_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        NodeId(id)
    }
    /// Get the current token and advance to the next one
    fn next(&mut self) -> Token {
        let token = match self.lexer.next() {
            Some(token) => token,
            None => Token {
                kind: LogosToken::Eof,
                span: Span::from(self.input.len()..self.input.len()),
            },
        };

        // Store the current token
        self.current = Some(token);
        token
    }

    fn current(&self) -> Token {
        match self.current {
            Some(token) => token,
            None => Token {
                kind: LogosToken::Eof,
                span: Span::from(self.input.len()..self.input.len()),
            },
        }
    }

    /// Eat token without getting its value
    fn advance(&mut self) {
        let _ = self.next();
    }

    fn skip_separator(&mut self) -> bool {
        let mut has_sep = false;
        while self.lexer.peek_is(LogosToken::Newline) || self.lexer.peek_is(LogosToken::Semicolon) {
            self.advance();
            has_sep = true;
        }
        has_sep
    }

    fn skip_newlines(&mut self) -> bool {
        let mut has_newline = false;
        while self.lexer.peek_is(LogosToken::Newline) {
            self.advance();
            has_newline = true;
        }
        has_newline
    }

    fn expect(&mut self, expected: LogosToken) -> ParseResult<Token> {
        let token = self.next();
        if token.kind == expected {
            Ok(token)
        } else {
            Err(ParseError::UnexpectedToken {
                expected: vec![format!("{expected:?}")],
                found: format!("{:?}", token.kind),
                span: token.span,
            })
        }
    }

    /// Enhanced expect that provides better error recovery
    fn expect_with_recovery(
        &mut self,
        expected: LogosToken,
        recovery_tokens: &[LogosToken],
    ) -> ParseResult<Token> {
        if let Some(peeked) = self.lexer.peek() {
            if peeked.kind == expected {
                Ok(self.next())
            } else if recovery_tokens.contains(&peeked.kind) {
                // Don't consume the recovery token, just return an error
                Err(ParseError::UnexpectedToken {
                    expected: vec![format!("{expected:?}")],
                    found: format!("{:?}", peeked.kind),
                    span: peeked.span,
                })
            } else {
                // Consume the unexpected token and continue
                let consumed = self.next();
                Err(ParseError::UnexpectedToken {
                    expected: vec![format!("{expected:?}")],
                    found: format!("{:?}", consumed.kind),
                    span: consumed.span,
                })
            }
        } else {
            // EOF case
            let span = self.lexer.span();
            Err(ParseError::UnexpectedEof {
                span: span.into(),
                expected: format!("{expected:?}"),
            })
        }
    }

    /// Create an error expression for recovery
    fn error_expr(&mut self, span: Span) -> Expr {
        Expr {
            id: self.get_node_id(),
            kind: ExprKind::Error,
            span,
        }
    }

    /// Create an error statement for recovery
    fn error_stmt(&mut self, span: Span) -> Stmt {
        Stmt {
            id: self.get_node_id(),
            kind: StmtKind::Error,
            span,
        }
    }

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let start_token = self.next();
        let start_pos = start_token.span.start;

        match start_token.kind {
            LogosToken::Int => {
                let text = start_token.text(self.input);
                let value = text.parse::<i64>().map_err(|_| ParseError::InvalidNumber {
                    span: start_token.span,
                    value: text.to_string(),
                })?;
                Ok(Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Literal(Literal::Int(value)),
                    span: start_token.span,
                })
            }
            LogosToken::Float => {
                let text = start_token.text(self.input);
                let value = text.parse::<f64>().map_err(|_| ParseError::InvalidNumber {
                    span: start_token.span,
                    value: text.to_string(),
                })?;
                Ok(Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Literal(Literal::Float(value)),
                    span: start_token.span,
                })
            }
            LogosToken::String => {
                let text = start_token.text(self.input);
                // Remove surrounding quotes and handle basic escape sequences
                let content = &text[1..text.len() - 1];
                Ok(Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Literal(Literal::Str(content.to_string())),
                    span: start_token.span,
                })
            }
            LogosToken::True => Ok(Expr {
                id: self.get_node_id(),
                kind: ExprKind::Literal(Literal::Int(1)),
                span: start_token.span,
            }),
            LogosToken::False => Ok(Expr {
                id: self.get_node_id(),
                kind: ExprKind::Literal(Literal::Int(0)),
                span: start_token.span,
            }),
            LogosToken::Ident => {
                let name = start_token.text(self.input).to_string();
                let spanned_name = Spanned::new(name.clone(), start_token.span);
                let ident = Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Identifier(name),
                    span: start_token.span,
                };

                // Handle function calls
                if self.lexer.peek_is(LogosToken::LParen) {
                    let opening_span = start_token.span;
                    self.advance();
                    self.push_delimiter(LogosToken::LParen, opening_span.into());

                    let args = self.parse_argument_list()?;

                    // Check for closing paren
                    if self.lexer.peek_is(LogosToken::RParen) {
                        let end_token = self.next();
                        self.pop_delimiter();

                        return Ok(Expr {
                            id: self.get_node_id(),
                            span: Span::from(ident.span.start..end_token.span.end),
                            kind: ExprKind::Call {
                                name: spanned_name,
                                args,
                            },
                        });
                    } else {
                        // Missing closing paren in function call
                        self.error(ParseError::MissingDelimiter {
                            span: self.current().span,
                            delimiter: ")".to_string(),
                            opening_span: opening_span,
                        });
                        self.pop_delimiter();

                        return Ok(Expr {
                            id: self.get_node_id(),
                            span: Span::from(ident.span.start..self.current().span.end),
                            kind: ExprKind::Call {
                                name: spanned_name,
                                args,
                            },
                        });
                    }
                }
                Ok(ident)
            }
            LogosToken::LParen => {
                let opening_span = start_token.span;
                self.push_delimiter(LogosToken::LParen, opening_span.into());
                self.skip_newlines();
                if self.lexer.peek_is(LogosToken::RParen) {
                    let end_token = self.next();
                    self.pop_delimiter();
                    return Ok(Expr {
                        id: self.get_node_id(),
                        kind: ExprKind::Unit,
                        span: Span::from(start_pos..end_token.span.end),
                    });
                }

                let expr = match self.parse_expression(0) {
                    Ok(expr) => expr,
                    Err(e) => {
                        self.error(e);
                        self.error_expr(start_token.span)
                    }
                };

                self.skip_newlines();

                // Check if we have a closing paren
                if self.lexer.peek_is(LogosToken::RParen) {
                    let end_token = self.next();
                    self.pop_delimiter();
                    Ok(Expr {
                        id: self.get_node_id(),
                        kind: expr.kind,
                        span: Span::from(start_pos..end_token.span.end),
                    })
                } else {
                    // Missing closing paren - report error but continue
                    self.error(ParseError::MissingDelimiter {
                        span: self.current().span,
                        delimiter: ")".to_string(),
                        opening_span: opening_span,
                    });
                    self.pop_delimiter();

                    // Return the expression we parsed, even without proper closing
                    Ok(Expr {
                        id: self.get_node_id(),
                        kind: expr.kind,
                        span: Span::from(start_pos..self.current().span.end),
                    })
                }
            }
            LogosToken::Not => {
                let operand = match self.parse_primary() {
                    Ok(expr) => expr,
                    Err(e) => {
                        self.error(e);
                        self.error_expr(start_token.span)
                    }
                };
                let end_pos = operand.span.end;
                Ok(Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    },
                    span: Span::from(start_pos..end_pos),
                })
            }
            LogosToken::Minus => {
                let operand = match self.parse_primary() {
                    Ok(expr) => expr,
                    Err(e) => {
                        self.error(e);
                        self.error_expr(start_token.span)
                    }
                };
                let end_pos = operand.span.end;
                Ok(Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        operand: Box::new(operand),
                    },
                    span: Span::from(start_pos..end_pos),
                })
            }
            LogosToken::LBrace => {
                self.push_delimiter(LogosToken::LBrace, start_token.span.into());
                let block = self.parse_block_content(start_pos)?;
                self.pop_delimiter();
                Ok(block)
            }
            _ => Err(ParseError::UnexpectedToken {
                span: start_token.span,
                found: format!("{:?}", start_token.kind),
                expected: vec![
                    "number".to_string(),
                    "string".to_string(),
                    "identifier".to_string(),
                    "(".to_string(),
                    "{".to_string(),
                    "!".to_string(),
                    "-".to_string(),
                ],
            }),
        }
    }

    /// Parse argument list with better error recovery
    fn parse_argument_list(&mut self) -> ParseResult<Vec<Expr>> {
        let mut args = Vec::new();

        self.skip_newlines();

        // Handle empty argument list
        if self.lexer.peek_is(LogosToken::RParen) {
            return Ok(args);
        }

        loop {
            // Parse argument
            match self.parse_expression(0) {
                Ok(arg) => args.push(arg),
                Err(e) => {
                    self.error(e);
                    // Try to recover by skipping to next comma or closing paren
                    self.skip_to_recovery_point(&[LogosToken::Comma, LogosToken::RParen]);

                    if self.lexer.peek_is(LogosToken::RParen) {
                        break;
                    }
                }
            }

            self.skip_newlines();
            let peek = self.lexer.peek();
            if let Some(peeked) = peek {
                match peeked.kind {
                    LogosToken::Comma => {
                        self.advance();
                        self.skip_newlines();
                        continue;
                    }
                    LogosToken::RParen => break,
                    _ => {
                        // Try to recover - if we see something that looks like it could be another argument,
                        // assume missing comma
                        if self.looks_like_expression_start() {
                            let peek_token = peeked;
                            self.error(ParseError::UnexpectedToken {
                                span: peek_token.span.into(),
                                found: format!("{:?}", peek_token.kind),
                                expected: vec!["','".to_string(), "')'".to_string()],
                            });
                            continue; // Assume missing comma
                        } else {
                            break;
                        }
                    }
                }
            } else {
                break;
            }
        }

        Ok(args)
    }

    /// Parse block content with better error recovery
    fn parse_block_content(&mut self, start_pos: usize) -> ParseResult<Expr> {
        let mut statements = Vec::new();
        let mut has_sep = true;
        let opening_span = (start_pos..start_pos + 1).into();

        self.skip_newlines();

        loop {
            // Check for closing brace
            if self.lexer.peek_is(LogosToken::RBrace) {
                let end_token = self.next();
                return Ok(Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Block(statements),
                    span: Span::from(start_pos..end_token.span.end),
                });
            }

            // Check for EOF
            if self.lexer.peek_is_eof() {
                let span = self.lexer.span();
                let e = ParseError::MissingDelimiter {
                    span: span,
                    delimiter: "}".to_string(),
                    opening_span,
                };
                self.error(e);
                return Ok(Expr {
                    id: self.get_node_id(),
                    kind: ExprKind::Block(statements),
                    span: Span::from(start_pos..self.current().span.end),
                });
            }

            // Parse statement
            let stmt = match self.parse_statement() {
                Ok(stmt) => stmt,
                Err(e) => {
                    self.error(e);
                    self.synchronize_in_block();
                    continue;
                }
            };

            // Check for missing separator
            if !statements.is_empty() && !has_sep {
                self.error(ParseError::MissingSeparator {
                    span: Span::from(stmt.span.start.saturating_sub(1)..stmt.span.start),
                });
            }

            statements.push(stmt);
            has_sep = self.skip_separator();
        }
    }

    /// Check if current token looks like the start of an expression
    fn looks_like_expression_start(&mut self) -> bool {
        if let Some(peeked) = self.lexer.peek() {
            matches!(
                peeked.kind,
                LogosToken::Int
                    | LogosToken::Float
                    | LogosToken::String
                    | LogosToken::True
                    | LogosToken::False
                    | LogosToken::Ident
                    | LogosToken::LParen
                    | LogosToken::LBrace
                    | LogosToken::Not
                    | LogosToken::Minus
            )
        } else {
            false
        }
    }

    /// Skip tokens until we find one of the recovery points
    fn skip_to_recovery_point(&mut self, recovery_tokens: &[LogosToken]) {
        while !self.lexer.peek_is_eof() {
            if let Some(peeked) = self.lexer.peek() {
                if recovery_tokens.contains(&peeked.kind) {
                    break;
                }
            }
            self.advance();
        }
    }

    /// Push delimiter onto stack for tracking
    fn push_delimiter(&mut self, delimiter: LogosToken, span: Span) {
        self.delimiter_stack.push((delimiter, span));
    }

    /// Pop delimiter from stack
    fn pop_delimiter(&mut self) {
        self.delimiter_stack.pop();
    }

    /// Enhanced synchronization for better error recovery
    fn synchronize(&mut self) {
        self.panic_mode = true;
        while !self.lexer.peek_is_eof() {
            if let Some(peeked) = self.lexer.peek() {
                match peeked.kind {
                    // Statement boundaries - good places to synchronize
                    LogosToken::KwLet
                    | LogosToken::KwFn
                    | LogosToken::KwReturn
                    | LogosToken::KwExtern => {
                        self.panic_mode = false;
                        break;
                    }
                    LogosToken::Semicolon | LogosToken::Newline => {
                        self.advance();
                        self.panic_mode = false;
                        break;
                    }
                    // If we're inside delimiters, try to find the matching closing delimiter
                    LogosToken::RBrace | LogosToken::RParen => {
                        // Only consume these if we have matching delimiters on the stack
                        if let Some((delimiter, _)) = self.delimiter_stack.last() {
                            let expected_closing = match delimiter {
                                LogosToken::LParen => LogosToken::RParen,
                                LogosToken::LBrace => LogosToken::RBrace,
                                _ => LogosToken::Eof, // shouldn't happen
                            };

                            if peeked.kind == expected_closing {
                                self.advance();
                                self.pop_delimiter();
                                if self.delimiter_stack.is_empty() {
                                    self.panic_mode = false;
                                    break;
                                }
                            } else {
                                // Found a closing delimiter but it doesn't match what we expect
                                break;
                            }
                        } else {
                            // Found a closing delimiter but we're not inside any delimiters
                            break;
                        }
                    }
                    _ => self.advance(),
                }
            } else {
                break;
            }
        }
    }

    /// Synchronize within a block (more conservative)
    fn synchronize_in_block(&mut self) {
        while !self.lexer.peek_is_eof() && !self.lexer.peek_is(LogosToken::RBrace) {
            if let Some(peeked) = self.lexer.peek() {
                match peeked.kind {
                    LogosToken::Semicolon | LogosToken::Newline => {
                        self.advance();
                        break;
                    }
                    _ => self.advance(),
                }
            } else {
                break;
            }
        }
    }

    /// Helper method to parse parameter lists with error recovery
    fn parse_parameter_list(&mut self) -> ParseResult<(Vec<Param>, Span)> {
        let opening_paren_span = if self.lexer.peek_is(LogosToken::LParen) {
            let token = self.next();
            self.push_delimiter(LogosToken::LParen, token.span.into());
            token.span
        } else {
            let span = self.lexer.span();
            let e = ParseError::UnexpectedToken {
                span: span.into(),
                found: format!("{:?}", self.lexer.peek_kind()),
                expected: vec!["(".to_string()],
            };
            self.error(e);
            span.into()
        };

        let mut params = Vec::new();
        self.skip_newlines();

        if !self.lexer.peek_is(LogosToken::RParen) {
            loop {
                // Parse parameter name
                let param_name = match self.expect_with_recovery(
                    LogosToken::Ident,
                    &[LogosToken::Colon, LogosToken::Comma, LogosToken::RParen],
                ) {
                    Ok(token) => token.text(self.input).to_string(),
                    Err(e) => {
                        self.error(e);
                        "_error_".to_string()
                    }
                };

                // Parse colon
                if let Err(e) = self.expect_with_recovery(
                    LogosToken::Colon,
                    &[LogosToken::Comma, LogosToken::RParen],
                ) {
                    self.error(e);
                }

                // Parse parameter type
                let param_type = self
                    .parse_type()
                    .unwrap_or(Spanned::new(Type::Unit, self.current().span));

                params.push((param_name, param_type));

                self.skip_newlines();

                if let Some(peeked) = self.lexer.peek() {
                    match peeked.kind {
                        LogosToken::Comma => {
                            self.advance();
                            self.skip_newlines();
                            continue;
                        }
                        LogosToken::RParen => break,
                        _ => {
                            let e = ParseError::UnexpectedToken {
                                span: peeked.span.into(),
                                found: format!("{:?}", peeked.kind),
                                expected: vec!["','".to_string(), "')'".to_string()],
                            };
                            self.error(e);
                            break;
                        }
                    }
                } else {
                    break;
                }
            }
        }

        // Parse closing paren
        if self.lexer.peek_is(LogosToken::RParen) {
            self.advance();
            self.pop_delimiter();
        } else {
            self.error(ParseError::MissingDelimiter {
                span: self.current().span.into(),
                delimiter: ")".to_string(),
                opening_span: opening_paren_span.into(),
            });
        }

        Ok((params, opening_paren_span.into()))
    }

    /// Helper method to parse function name with error recovery
    fn parse_function_name(&mut self) -> ParseResult<Spanned<String>> {
        match self.expect_with_recovery(
            LogosToken::Ident,
            &[LogosToken::LParen, LogosToken::RParen, LogosToken::Newline],
        ) {
            Ok(token) => Ok(Spanned::new(token.text(self.input).to_string(), token.span)),
            Err(e) => {
                self.error(e);
                Ok(Spanned::new("_error_".to_string(), self.current().span))
            }
        }
    }

    fn parse_type(&mut self) -> Option<Spanned<Type>> {
        let start_pos = self.lexer.peek()?.span.start;

        // Handle function types: fn(param_types...) -> return_type
        if self.lexer.peek_is(LogosToken::KwFn) {
            self.advance(); // consume 'fn'

            // Parse parameter types
            if !self.lexer.peek_is(LogosToken::LParen) {
                self.error(ParseError::UnexpectedToken {
                    span: self.current().span,
                    found: format!("{:?}", self.current().kind),
                    expected: vec!["(".to_string()],
                });
                return None;
            }

            self.advance(); // consume '('
            let mut param_types = Vec::new();

            if !self.lexer.peek_is(LogosToken::RParen) {
                loop {
                    if let Some(param_type) = self.parse_type() {
                        param_types.push(param_type);
                    } else {
                        break;
                    }

                    if self.lexer.peek_is(LogosToken::Comma) {
                        self.advance();
                        continue;
                    } else if self.lexer.peek_is(LogosToken::RParen) {
                        break;
                    } else {
                        self.error(ParseError::UnexpectedToken {
                            span: self.current().span,
                            found: format!("{:?}", self.current().kind),
                            expected: vec!["','".to_string(), "')'".to_string()],
                        });
                        break;
                    }
                }
            }

            if !self.lexer.peek_is(LogosToken::RParen) {
                self.error(ParseError::UnexpectedToken {
                    span: self.current().span,
                    found: format!("{:?}", self.current().kind),
                    expected: vec![")".to_string()],
                });
                return None;
            }
            self.advance(); // consume ')'

            // Parse return type
            let return_type = if self.lexer.peek_is(LogosToken::Arrow) {
                self.advance(); // consume '->'
                self.parse_type().unwrap_or_else(|| {
                    let span = self.current().span;
                    Spanned::new(Type::Unit, span)
                })
            } else {
                let span = self.current().span;
                Spanned::new(Type::Unit, span)
            };

            let end_pos = self.current().span.end;
            return Some(Spanned::new(
                Type::Function {
                    params: param_types,
                    return_type: Box::new(return_type),
                },
                Span::from(start_pos..end_pos),
            ));
        }

        if self.lexer.peek_is(LogosToken::LParen) {
            self.advance();

            if !self.lexer.peek_is(LogosToken::RParen) {
                self.error(ParseError::UnexpectedToken {
                    span: self.current().span,
                    found: format!("{:?}", self.current().kind),
                    expected: vec![")".to_string()],
                });
                return None;
            }
            self.advance();
            return Some(Spanned::new(
                Type::Unit,
                Span::from(start_pos..self.current().span.end),
            ));
        }

        // Handle basic types
        match self.expect(LogosToken::Ident) {
            Ok(token) => {
                let type_name = token.text(self.input);
                let span = token.span;
                let type_variant = match type_name {
                    "i8" => Type::I8,
                    "i16" => Type::I16,
                    "i32" => Type::I32,
                    "i64" => Type::I64,
                    "str" => Type::Str,
                    "_" => Type::Infer,
                    _ => {
                        self.error(ParseError::UnexpectedToken {
                            span: token.span.into(),
                            found: type_name.to_string(),
                            expected: vec![
                                "i8".to_string(),
                                "i16".to_string(),
                                "i32".to_string(),
                                "i64".to_string(),
                                "str".to_string(),
                                "fn".to_string(),
                            ],
                        });
                        return None;
                    }
                };
                Some(Spanned::new(type_variant, span))
            }
            Err(e) => {
                self.error(e);
                None
            }
        }
    }

    pub fn parse_statement(&mut self) -> ParseResult<Stmt> {
        let start = if let Some(peeked) = self.lexer.peek() {
            peeked.span.start
        } else {
            self.lexer.span().start
        };

        if let Some(peeked) = self.lexer.peek() {
            match peeked.kind {
                LogosToken::KwLet => {
                    self.advance();

                    // Parse identifier with error recovery
                    let name = match self.expect_with_recovery(
                        LogosToken::Ident,
                        &[LogosToken::Eq, LogosToken::Semicolon, LogosToken::Newline],
                    ) {
                        Ok(token) => Spanned::new(token.text(self.input).to_string(), token.span),
                        Err(e) => {
                            self.error(e);
                            Spanned::new("_error_".to_string(), self.current().span)
                        }
                    };

                    // Parse optional type annotation
                    let type_ann = if self.lexer.peek_is(LogosToken::Colon) {
                        self.advance();
                        self.parse_type()
                    } else {
                        None
                    };

                    // Parse equals sign with error recovery
                    if let Err(e) = self.expect_with_recovery(
                        LogosToken::Eq,
                        &[
                            LogosToken::Semicolon,
                            LogosToken::Newline,
                            LogosToken::RBrace,
                        ],
                    ) {
                        self.error(e);
                        // If we don't have an equals sign, create a default value
                        let value = self.error_expr(self.current().span);
                        return Ok(Stmt {
                            id: self.get_node_id(),
                            kind: StmtKind::Let {
                                name,
                                type_ann,
                                value,
                            },
                            span: Span::from(start..self.current().span.end),
                        });
                    }

                    // Parse the value expression
                    let value = match self.parse_expression(0) {
                        Ok(expr) => expr,
                        Err(e) => {
                            self.error(e);
                            self.error_expr(self.current().span)
                        }
                    };

                    Ok(Stmt {
                        id: self.get_node_id(),
                        kind: StmtKind::Let {
                            name,
                            type_ann,
                            value: value.clone(),
                        },
                        span: Span::from(start..value.span.end),
                    })
                }

                LogosToken::KwFn => {
                    self.advance();

                    // Parse function name
                    let name = self.parse_function_name()?;

                    // Parse parameter list
                    let (params, _) = self.parse_parameter_list()?;

                    // Parse return type
                    let return_type = if self.lexer.peek_is(LogosToken::Arrow) {
                        self.advance();
                        self.parse_type().unwrap_or(Spanned::new(
                            Type::Unit,
                            (self.current().span.start + 1..self.current().span.end + 1).into(),
                        ))
                    } else {
                        Spanned::new(
                            Type::Unit,
                            (self.current().span.start + 1..self.current().span.end + 1).into(),
                        )
                    };

                    // Parse function body
                    let body = if self.lexer.peek_is(LogosToken::LBrace) {
                        match self.parse_primary() {
                            Ok(expr) => expr,
                            Err(e) => {
                                self.error(e);
                                self.error_expr(self.current().span)
                            }
                        }
                    } else {
                        let span = self.lexer.span();
                        let e = ParseError::UnexpectedToken {
                            span: span.into(),
                            found: format!("{:?}", self.lexer.peek_kind()),
                            expected: vec!["{".to_string()],
                        };
                        self.error(e);
                        self.error_expr(self.current().span)
                    };

                    Ok(Stmt {
                        id: self.get_node_id(),
                        kind: StmtKind::Function {
                            name,
                            params,
                            return_type,
                            body: body.clone(),
                        },
                        span: Span::from(start..body.span.end),
                    })
                }

                LogosToken::KwExtern => {
                    self.advance();
                    self.expect(LogosToken::KwFn)?;
                    let name = self.parse_function_name()?;
                    let (params, _) = self.parse_parameter_list()?;
                    // Parse return type
                    let return_type = if self.lexer.peek_is(LogosToken::Arrow) {
                        self.advance();
                        self.parse_type()
                            .unwrap_or(Spanned::new(Type::Unit, self.current().span))
                    } else {
                        Spanned::new(Type::Unit, self.current().span)
                    };
                    Ok(Stmt {
                        id: self.get_node_id(),
                        kind: StmtKind::Extern {
                            name,
                            params,
                            return_type,
                        },
                        span: Span::from(start..self.current().span.end),
                    })
                }

                LogosToken::KwReturn => {
                    self.advance();

                    // Parse optional return value
                    let value = if self.looks_like_expression_start() {
                        match self.parse_expression(0) {
                            Ok(expr) => Some(expr),
                            Err(e) => {
                                self.error(e);
                                Some(self.error_expr(self.current().span))
                            }
                        }
                    } else {
                        None
                    };

                    let end_pos = value
                        .as_ref()
                        .map(|v| v.span.end)
                        .unwrap_or(self.current().span.end);

                    Ok(Stmt {
                        id: self.get_node_id(),
                        kind: StmtKind::Return(value),
                        span: Span::from(start..end_pos),
                    })
                }

                _ => {
                    let expr = self.parse_expression(0)?;
                    let span = expr.span;
                    Ok(Stmt {
                        id: self.get_node_id(),
                        kind: StmtKind::Expr(expr),
                        span,
                    })
                }
            }
        } else {
            // EOF case
            let span = self.lexer.span();
            Err(ParseError::UnexpectedEof {
                span: span.into(),
                expected: "statement".to_string(),
            })
        }
    }

    pub fn parse_expression(&mut self, min_prec: u8) -> ParseResult<Expr> {
        let mut left = self.parse_primary()?;
        loop {
            let op = if let Some(peeked) = self.lexer.peek() {
                match peeked.kind.to_infix_op() {
                    Some(x) => x,
                    None => break,
                }
            } else {
                break;
            };

            let (left_prec, right_prec) = op.precedence();
            if left_prec < min_prec {
                break;
            }
            self.advance();

            let right = match self.parse_expression(right_prec) {
                Ok(expr) => expr,
                Err(e) => {
                    self.error(e);
                    // Create error expression and continue
                    self.error_expr(self.current().span)
                }
            };

            let start_pos = left.span.start;
            let end_pos = right.span.end;
            left = Expr {
                id: self.get_node_id(),
                kind: ExprKind::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                span: Span::from(start_pos..end_pos),
            };
        }

        Ok(left)
    }

    fn error(&mut self, error: ParseError) {
        // Only add error if we're not already in panic mode
        // This prevents cascading errors
        if !self.panic_mode {
            self.errors.push(error);
        }
    }

    /// Parse a complete program (multiple statements)
    pub fn parse_program(mut self) -> (Vec<Stmt>, Vec<ParseError>) {
        let mut statements = Vec::new();
        let mut has_sep = true;

        while !self.lexer.peek_is_eof() {
            self.skip_newlines();
            if self.lexer.peek_is_eof() {
                break;
            }

            let span_start = self.current().span.start;
            let stmt = match self.parse_statement() {
                Ok(stmt) => {
                    self.panic_mode = false; // Successfully parsed a statement
                    stmt
                }
                Err(e) => {
                    self.error(e);
                    self.synchronize();

                    // Create error statement
                    self.error_stmt(Span::from(span_start..self.current().span.end))
                }
            };

            // Check for missing separator (but only if not in panic mode)
            if !statements.is_empty() && !has_sep && !self.panic_mode {
                self.error(ParseError::MissingSeparator {
                    span: Span::from(stmt.span.start.saturating_sub(1)..stmt.span.start),
                });
            }

            statements.push(stmt);
            has_sep = self.skip_separator();
        }

        // Check for unclosed delimiters
        while let Some((delimiter, span)) = self.delimiter_stack.pop() {
            let closing = match delimiter {
                LogosToken::LParen => ")",
                LogosToken::LBrace => "}",
                _ => "delimiter",
            };

            self.errors.push(ParseError::MissingDelimiter {
                span: (self.input.len()..self.input.len()).into(),
                delimiter: closing.to_string(),
                opening_span: span,
            });
        }

        (statements, self.errors)
    }
}

impl BinOp {
    /// Returns (left_precedence, right_precedence) for the operator
    fn precedence(&self) -> (u8, u8) {
        match self {
            BinOp::Or => (1, 2),
            BinOp::And => (3, 4),
            BinOp::Eq | BinOp::Neq => (5, 6),
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => (7, 8),
            BinOp::Add | BinOp::Sub => (9, 10),
            BinOp::Mul | BinOp::Div => (11, 12),
        }
    }
}
