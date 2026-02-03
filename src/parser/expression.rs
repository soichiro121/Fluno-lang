// src/parser/expression.rs

use crate::ast::node::{
    BinaryOp, Expression, Identifier, Literal, Precedence, Span, UnaryOp, Path, PathSeg
};
use crate::lexer::{TokenKind};
use crate::parser::{ParseError, ParseResult, Parser};

impl<'a> Parser<'a> {
    pub fn parse_expression_with_precedence(
        &mut self,
        precedence: Precedence,
    ) -> ParseResult<Expression> {
        let mut left = self.parse_prefix()?;

        loop {
            loop {
                if let Expression::Variable { name, span } = &left {
                    if self.check(TokenKind::LBrace) {
                        let is_type_name = if let Some(last) = name.last_ident() {
                            last.name.chars().next().map_or(false, |c| c.is_uppercase())
                        } else {
                            false
                        };

                        if is_type_name {
                            self.advance()?;
                            
                            let mut fields = Vec::new();
                            while !self.check(TokenKind::RBrace) && !self.is_at_end() {
                                let field_name_tok = self.expect(TokenKind::Identifier)?;
                                let field_name = Identifier::new(
                                    field_name_tok.text.unwrap_or_default(),
                                    Span::new(field_name_tok.line, field_name_tok.column, 0),
                                );

                                let value = if self.match_any(&[TokenKind::Colon]) {
                                    self.parse_expression_with_precedence(Precedence::Lowest)?
                                } else {
                                    let var_path = Path::from_ident(field_name.clone());
                                    Expression::Variable {
                                        name: var_path,
                                        span: field_name.span,
                                    }
                                };

                                fields.push(crate::ast::node::FieldInit {
                                    name: field_name,
                                    value,
                                    span: Span::initial(),
                                });

                                if !self.match_any(&[TokenKind::Comma]) {
                                    break;
                                }
                            }
                            self.expect(TokenKind::RBrace)?;

                            let flat_name_str = name.to_string(); 
                            let struct_name_ident = Identifier::new(flat_name_str, span.clone());

                            left = Expression::Struct {
                                name: struct_name_ident,
                                fields,
                                span: span.clone(), 
                            };
                            continue;
                        }
                    }
                }

                if self.check(TokenKind::LParen) {
                    left = self.parse_call_expression(left)?;
                    continue;
                }
                
                if self.check(TokenKind::LBracket) {
                    left = self.parse_index_expression(left)?;
                    continue;
                }

                if self.check(TokenKind::Dot) {
                    left = self.parse_dot_postfix(left)?;
                    continue;
                }

                if self.check(TokenKind::Question) {
                    let start_span = left.span();
                    self.advance()?; 
                    let span = Span::new(start_span.line, start_span.column, 0);
                    left = Expression::Try {
                        expr: Box::new(left),
                        span,
                    };
                    continue;
                }

                break;
            }

            if self.is_at_end() || self.check(TokenKind::Semicolon) {
                break;
            }

            if precedence > self.current_precedence() {
                break;
            }

            if self.check(TokenKind::DotDot) {
                if precedence > Precedence::Assignment {
                    break;
                }
                left = self.parse_range_expression(left)?;
                continue;
            }

            if let Some(op) = self.current_binary_op() {
                let op_precedence = self.current_precedence();
                left = self.parse_infix(left, op, op_precedence)?;
                continue;
            } else {
                break;
            }
        }

        Ok(left)
    }


    fn parse_dot_postfix(&mut self, left: Expression) -> ParseResult<Expression> {
        let _ = self.advance()?; 

        let startspan = left.span();

        if self.check(TokenKind::Await) {
            let _await_tok = self.advance()?; 
            let span = Span::new(startspan.line, startspan.column, 0); 
            return Ok(Expression::Await {
                expr: Box::new(left),
                span,
            });
        }

        let name = self.parse_identifier_helper()?; 
        let span = Span::new(startspan.line, startspan.column, 0);

        if self.check(TokenKind::LParen) {
            let _ = self.advance()?;
            let mut args = Vec::new();
            if !self.check(TokenKind::RParen) {
                loop {
                    args.push(self.parse_expression_with_precedence(Precedence::Lowest)?);
                    if !self.match_any(&[TokenKind::Comma]) {
                        break;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;

            return Ok(Expression::MethodCall {
                receiver: Box::new(left),
                method: name,
                args,
                span,
                resolved: None, 
            });
        }

        Ok(Expression::FieldAccess {
            object: Box::new(left),
            field: name,
            span,
        })
    }

    fn parse_infix(
        &mut self,
        left: Expression,
        op: BinaryOp,
        precedence: Precedence,
    ) -> ParseResult<Expression> {
        let op_token = self.current.kind;

        let _ = self.advance()?; 

        let prec_u8 = precedence as u8;
        let min_prec_u8 = if op_token.is_right_associative() {
            prec_u8
        } else {
            prec_u8 + 1
        };

        let next_prec: Precedence = unsafe { std::mem::transmute(min_prec_u8) };

        let right = self.parse_expression_with_precedence(next_prec)?;

        let left_span = left.span();
        let span = Span::new(left_span.line, left_span.column, 0);

        Ok(Expression::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
            span,
        })
    }



    fn parse_range_expression(&mut self, left: Expression) -> ParseResult<Expression> {
        let _ = self.advance()?; // Consume '..'
        let start_span = left.span();
        
        let end = if self.is_at_end() || 
                     self.check(TokenKind::Semicolon) || 
                     self.check(TokenKind::Comma) || 
                     self.check(TokenKind::RBrace) || 
                     self.check(TokenKind::RParen) || 
                     self.check(TokenKind::RBracket) {
            None
        } else {
            Some(self.parse_expression_with_precedence(Precedence::Lowest)?)
        };
        
        // Approximate span (todo: real span union)
        let span = Span::new(start_span.line, start_span.column, 0); 

        Ok(Expression::Range {
            start: Some(Box::new(left)),
            end: end.map(Box::new),
            inclusive: false,
            span,
        })
    }

    fn _parse_field_access(&mut self, left: Expression) -> ParseResult<Expression> {
        let _ = self.advance(); 
        let start_span = left.span();
        
        let name = self.parse_identifier_helper()?;

        let span = Span::new(start_span.line, start_span.column, 0);

        Ok(Expression::FieldAccess {
            object: Box::new(left),
            field: name,
            span,
        })
    }
    
    
    fn _parse_struct_literal(&mut self, name: Identifier, span: Span) -> ParseResult<Expression> {
        self.expect(TokenKind::LBrace)?; 
        let mut fields = Vec::new();
        
        if !self.check(TokenKind::RBrace) {
            loop {
                let field_name = self.parse_identifier_helper()?;
                self.expect(TokenKind::Colon)?;
                let value = self.parse_expression_with_precedence(Precedence::Lowest)?;
                
                fields.push(crate::ast::node::FieldInit {
                    name: field_name,
                    value,
                    span: Span::initial(),
                });
                
                if !self.match_any(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        self.expect(TokenKind::RBrace)?;
        
        Ok(Expression::Struct {
            name,
            fields,
            span,
        })
    }


    fn parse_prefix(&mut self) -> ParseResult<Expression> {
        let token = self.current.clone();
        let span = Span::new(
            token.line,
            token.column,
            token.text.as_ref().map(|s| s.len()).unwrap_or(1),
        );

        match &token.kind {
            TokenKind::IntLiteral => {
                let _ = self.advance();
                let text = token.text.as_ref().ok_or(ParseError::UnexpectedEof {
                    expected: "integer literal".into(),
                    line: token.line,
                    column: token.column,
                })?;
                let value = text.parse::<i64>().map_err(|_| ParseError::InvalidSyntax {
                    message: format!("Invalid integer literal: {}", text),
                    line: token.line,
                    column: token.column,
                })?;

                Ok(Expression::Literal {
                    value: Literal::Int(value),
                    span,
                })
            }

            TokenKind::SelfLower => {
                let _ = self.advance(); 
                Ok(Expression::Variable {
                    name: Path {
                        segments: vec![PathSeg::Ident(Identifier::new("self".to_string(), span.clone()))],
                        span: span.clone(),
                        resolved: None,
                    },
                    span,
                })
            }

            TokenKind::FloatLiteral => {
                let _ = self.advance();
                let text = token.text.as_ref().ok_or(ParseError::UnexpectedEof {
                    expected: "float literal".into(),
                    line: token.line,
                    column: token.column,
                })?;
                let value = text.parse::<f64>().map_err(|_| ParseError::InvalidSyntax {
                    message: format!("Invalid float literal: {}", text),
                    line: token.line,
                    column: token.column,
                })?;

                Ok(Expression::Literal {
                    value: Literal::Float(value),
                    span,
                })
            }

            TokenKind::StringLiteral => {
                let _ = self.advance();
                let text = token.text.clone().unwrap_or_default();
                let content = if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') {
                    text[1..text.len() - 1].to_string()
                } else {
                    text
                };

                if content.contains('{') && content.contains('}') {
                    return self.desugar_string_interpolation(&content, span);
                }

                Ok(Expression::Literal {
                    value: Literal::String(content),
                    span,
                })
            }

            TokenKind::True => {
                let _ = self.advance();
                Ok(Expression::Literal {
                    value: Literal::Bool(true),
                    span,
                })
            }

            TokenKind::False => {
                let _ = self.advance();
                Ok(Expression::Literal {
                    value: Literal::Bool(false),
                    span,
                })
            }

            TokenKind::With => self.parse_with_expression(),

            TokenKind::Identifier => {
                let name_ref = token.text.as_ref().map(|s| s.as_str()).unwrap_or("");

                match name_ref {
                    "Some" => {
                        if self.check(TokenKind::LParen) {
                            let _ = self.advance();
                            self.expect(TokenKind::LParen)?;
                            let inner_expr =
                                self.parse_expression_with_precedence(Precedence::Lowest)?;
                            self.expect(TokenKind::RParen)?;

                            Ok(Expression::Some {
                                expr: Box::new(inner_expr),
                                span,
                            })
                        } else {
                            self.parse_identifier_expression()
                        }
                    }

                    "None" => {
                        let _ = self.advance();
                        Ok(Expression::None { span })
                    }

                    "Ok" => {
                        if self.check(TokenKind::LParen) {
                            let _ = self.advance();
                            self.expect(TokenKind::LParen)?;
                            let inner_expr =
                                self.parse_expression_with_precedence(Precedence::Lowest)?;
                            self.expect(TokenKind::RParen)?;
                            Ok(Expression::Ok {
                                expr: Box::new(inner_expr),
                                span,
                            })
                        } else {
                            self.parse_identifier_expression()
                        }
                    }

                    "Err" => {
                        if self.check(TokenKind::LParen) {
                            let _ = self.advance();
                            self.expect(TokenKind::LParen)?;
                            let inner_expr =
                                self.parse_expression_with_precedence(Precedence::Lowest)?;
                            self.expect(TokenKind::RParen)?;
                            Ok(Expression::Err {
                                expr: Box::new(inner_expr),
                                span,
                            })
                        } else {
                            self.parse_identifier_expression()
                        }
                    }

                    _ => {
                        let enum_name = self.parse_identifier_helper()?; 
                        let enum_span = enum_name.span;

                        if self.check(TokenKind::ColonColon) {
                            let _ = self.advance()?; 

                            let vtok = self.expect(TokenKind::Identifier)?;
                            let variant = Identifier::new(
                                vtok.text.unwrap_or_default(),
                                Span::new(vtok.line, vtok.column, 0),
                            );

                            let islower = variant
                                .name
                                .chars()
                                .next()
                                .map(|c| c.is_lowercase())
                                .unwrap_or(false);

                            if islower {
                                let traitpath = Path {
                                    segments: vec![PathSeg::Ident(enum_name.clone())],
                                    span: enum_span,
                                    resolved: None,
                                };

                                let ufcs_expr = Expression::UfcsMethod {
                                    trait_path: traitpath,
                                    method: variant,
                                    span: enum_span,
                                };

                                if self.check(TokenKind::LParen) {
                                    let _ = self.advance()?; 
                                    let mut args = Vec::new();
                                    if !self.check(TokenKind::RParen) {
                                        loop {
                                            args.push(self.parse_expression_with_precedence(Precedence::Lowest)?);
                                            if !self.match_any(&[TokenKind::Comma]) {
                                                break;
                                            }
                                        }
                                    }
                                    self.expect(TokenKind::RParen)?;

                                    return Ok(Expression::Call {
                                        callee: Box::new(ufcs_expr),
                                        args,
                                        span: enum_span,
                                    });
                                }

                                return Ok(ufcs_expr);
                            }

                            if self.check(TokenKind::LParen) {
                                let mut args = Vec::new();
                                let _ = self.advance()?; 

                                if !self.check(TokenKind::RParen) {
                                    loop {
                                        args.push(self.parse_expression_with_precedence(Precedence::Lowest)?);
                                        if !self.match_any(&[TokenKind::Comma]) {
                                            break;
                                        }
                                    }
                                }

                                self.expect(TokenKind::RParen)?;

                                return Ok(Expression::Enum {
                                    name: enum_name,
                                    variant,
                                    args,
                                    named_fields: None,
                                    span: enum_span,
                                });
                            }

                            if self.check(TokenKind::LBrace) {
                                let _ = self.advance()?; 

                                let mut args: Vec<Expression> = Vec::new();

                                if !self.check(TokenKind::RBrace) {
                                    loop {
                                        let _fieldnametok = self.expect(TokenKind::Identifier)?;
                                        self.expect(TokenKind::Colon)?;
                                        let value = self.parse_expression_with_precedence(Precedence::Lowest)?;
                                        args.push(value);

                                        if !self.match_any(&[TokenKind::Comma]) {
                                            break;
                                        }
                                    }
                                }

                                self.expect(TokenKind::RBrace)?;

                                return Ok(Expression::Enum {
                                    name: enum_name,
                                    variant,
                                    args,
                                    named_fields: None,
                                    span: enum_span,
                                });
                            }

                            return Ok(Expression::Enum {
                                name: enum_name,
                                variant,
                                args: Vec::new(),
                                named_fields: None,
                                span: enum_span,
                            });
                        }

                        let path = Path {
                            segments: vec![PathSeg::Ident(enum_name)],
                            span: enum_span,
                            resolved: None,
                        };
                        Ok(Expression::Variable {
                            name: path,
                            span: enum_span,
                        })

                    }
                }
            }

            TokenKind::Minus => self.parse_unary(UnaryOp::Neg),
            TokenKind::Not => self.parse_unary(UnaryOp::Not),
            TokenKind::Match => self.parse_match_expression(),

            TokenKind::LParen => {
                let _ = self.advance()?; 

                if self.check(TokenKind::RParen) {
                    let _ = self.advance()?; 
                    return Ok(Expression::Literal {
                        value: Literal::Unit,
                        span,
                    });
                }

                let expr = self.parse_expression_with_precedence(Precedence::Lowest)?;
                self.expect(TokenKind::RParen)?;
                Ok(Expression::Paren {
                    expr: Box::new(expr),
                    span,
                })
            }

            TokenKind::LBrace => {
                let block = self.parse_block()?;
                Ok(Expression::Block { block, span })
            }

            TokenKind::If => self.parse_if_expression(),

            TokenKind::LBracket => self.parse_array_literal(),

            TokenKind::Spawn => {
                let _ = self.advance()?;
                let inner_expr = self.parse_expression_with_precedence(Precedence::Lowest)?;
                Ok(Expression::Spawn {
                    expr: Box::new(inner_expr),
                    span,
                })
            }

            TokenKind::BitOr | TokenKind::Or => self.parse_lambda_expression(),

            _ => Err(ParseError::UnexpectedToken {
                expected: TokenKind::Identifier,
                found: token.kind.clone(),
                line: token.line,
                column: token.column,
            }),
        }
    }

    fn desugar_string_interpolation(&self, content: &str, span: Span) -> ParseResult<Expression> {
        let mut exprs: Vec<Expression> = Vec::new();
        let mut last_pos = 0;
        let len = content.len();
        let bytes = content.as_bytes(); 
        
        let mut i = 0;
        while i < len {
            if bytes[i] == b'{' {
                if let Some(close_pos) = content[i..].find('}') {
                    let abs_close = i + close_pos;
                    let inner = &content[i+1..abs_close];
                    let inner_trimmed = inner.trim();
                    
                    if !inner_trimmed.is_empty() && inner_trimmed.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        if i > last_pos {
                            exprs.push(Expression::Literal {
                                value: Literal::String(content[last_pos..i].to_string()),
                                span, 
                            });
                        }
                        exprs.push(Expression::Variable {
                            name: Path {
                                segments: vec![PathSeg::Ident(Identifier::new(inner_trimmed.to_string(), span))],
                                span,
                                resolved: None
                            },
                            span,
                        });
                        last_pos = abs_close + 1;
                        i = last_pos;
                        continue;
                    }
                }
            }
            i += 1;
        }
        
        if last_pos < len {
            exprs.push(Expression::Literal {
                value: Literal::String(content[last_pos..].to_string()),
                span,
            });
        }
        
        if exprs.is_empty() {
            return Ok(Expression::Literal { value: Literal::String("".to_string()), span });
        }
        
        let mut combined = exprs.remove(0);
        for expr in exprs {
            combined = Expression::Binary {
                left: Box::new(combined),
                op: BinaryOp::Add,
                right: Box::new(expr),
                span,
            };
        }
        
        Ok(combined)
    }


    
    
    fn parse_lambda_expression(&mut self) -> ParseResult<Expression> {
        let start_span = Span::new(self.current.line, self.current.column, 1);

        if self.match_any(&[TokenKind::Or]) {
            let params = Vec::new();
            let body = if self.check(TokenKind::LBrace) {
                let block = self.parse_block()?;
                Box::new(Expression::Block { 
                    block, 
                    span: Span::initial() 
                })
            } else {
                Box::new(self.parse_expression_with_precedence(Precedence::Lowest)?)
            };

            return Ok(Expression::Lambda {
                params,
                body,
                span: start_span,
            });
        }

        self.expect(TokenKind::BitOr)?; 

        let mut params = Vec::new();
        
        if !self.check(TokenKind::BitOr) {
            loop {
                let name = self.parse_identifier_helper()?;
                let ty = if self.match_any(&[TokenKind::Colon]) {
                    self.parse_type()?
                } else {
                    crate::ast::node::Type::Infer
                };
                
                params.push(crate::ast::node::Parameter {
                    name,
                    ty,
                    span: Span::initial(),
                });

                if !self.match_any(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        
        self.expect(TokenKind::BitOr)?; 

        let body = if self.check(TokenKind::LBrace) {
            let block = self.parse_block()?;
            Box::new(Expression::Block { 
                block, 
                span: Span::initial() 
            })
        } else {
            Box::new(self.parse_expression_with_precedence(Precedence::Lowest)?)
        };

        Ok(Expression::Lambda {
            params,
            body,
            span: start_span,
        })
    }




    fn parse_with_expression(&mut self) -> ParseResult<Expression> {
        let start_token = self.current.clone();
        let start_span = Span::new(start_token.line, start_token.column, 4);

        let _ = self.advance(); 
        let name = self.parse_identifier_helper()?;
        self.expect(TokenKind::Assign)?;
        let initializer = Box::new(self.parse_expression_with_precedence(Precedence::Lowest)?);
        let body = self.parse_block()?;

        Ok(Expression::With {
            name,
            initializer,
            body,
            span: start_span,
        })
    }

    fn parse_identifier_expression(&mut self) -> ParseResult<Expression> {
        let ident = self.parse_identifier_helper()?;
        let span = ident.span;

        let path = Path {
            segments: vec![PathSeg::Ident(ident)],
            span,
            resolved: None,
        };

        Ok(Expression::Variable { name: path, span })
    }


    fn parse_identifier_helper(&mut self) -> ParseResult<Identifier> {
        let token;
        
        if self.check(TokenKind::Identifier) {
            token = self.current.clone(); 
            let _ = self.advance()?;              
        } else if self.check(TokenKind::SelfLower) {
            token = self.current.clone(); 
            let _ = self.advance()?;              
        } else {
            return Err(ParseError::UnexpectedToken {
                expected: TokenKind::Identifier,
                found: self.current.kind.clone(),
                line: self.current.line,
                column: self.current.column,
            });
        }

        let name = token.text.clone().unwrap_or_else(|| "self".to_string());
        let span = Span::new(token.line, token.column, name.len());
        Ok(Identifier::new(name, span))
    }


    fn parse_unary(&mut self, op: UnaryOp) -> ParseResult<Expression> {
        let token = self.current.clone();
        let start_span = Span::new(token.line, token.column, 1);
        
        let _ = self.advance(); 
        let operand = self.parse_expression_with_precedence(Precedence::Prefix)?;

        Ok(Expression::Unary {
            op,
            operand: Box::new(operand),
            span: start_span,
        })
    }


    fn parse_call_expression(&mut self, callee: Expression) -> ParseResult<Expression> {
        if let Expression::Variable { name, span } = &callee {
            match name.last_name().unwrap() {
                "Some" => {
                    let _ = self.advance(); 
                    let inner = self.parse_expression_with_precedence(Precedence::Lowest)?;
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expression::Some {
                        expr: Box::new(inner),
                        span: *span,
                    });
                }
                "Ok" => {
                    let _ = self.advance(); 
                    let inner = self.parse_expression_with_precedence(Precedence::Lowest)?;
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expression::Ok {
                        expr: Box::new(inner),
                        span: *span,
                    });
                }
                "Err" => {
                    let _ = self.advance(); 
                    let inner = self.parse_expression_with_precedence(Precedence::Lowest)?;
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expression::Err {
                        expr: Box::new(inner),
                        span: *span,
                    });
                }
                "None" => {
                    let _ = self.advance(); 
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expression::None { span: *span });
                }
                _ => {}
            }
        }

        let _ = self.advance();
        let start_span = callee.span();
        let mut args = Vec::new();

        if !self.check(TokenKind::RParen) {
            loop {
                args.push(self.parse_expression_with_precedence(Precedence::Lowest)?);
                if !self.match_any(&[TokenKind::Comma]) {
                    break;
                }
            }
        }

        self.expect(TokenKind::RParen)?;

        Ok(Expression::Call {
            callee: Box::new(callee),
            args,
            span: start_span,
        })
    }

    fn parse_index_expression(&mut self, left: Expression) -> ParseResult<Expression> {
        let _ = self.advance()?;
        let start_span = left.span();

        let index = self.parse_expression_with_precedence(Precedence::Lowest)?;
        self.expect(TokenKind::RBracket)?;

        Ok(Expression::Index {
            object: Box::new(left),
            index: Box::new(index),
            span: start_span,
        })
    }

    pub fn parse_if_expression(&mut self) -> ParseResult<Expression> {
        let token = self.current.clone();
        let start_span = Span::new(token.line, token.column, 2);

        let _ = self.advance()?;
        let condition = self.parse_expression_with_precedence(Precedence::Lowest)?;
        let then_branch = self.parse_block()?;

        let else_branch = if self.check(TokenKind::Else) {
            let _ = self.advance()?;
            if self.check(TokenKind::If) {
                let if_expr = self.parse_if_expression()?;
                Some(crate::ast::node::Block {
                    statements: vec![crate::ast::node::Statement::Expression(if_expr)],
                    span: Span::initial(),
                })
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };

        Ok(Expression::If {
            condition: Box::new(condition),
            then_branch,
            else_branch,
            span: start_span,
        })
    }
}
