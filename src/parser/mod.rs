// src/parser/mod.rs

mod error;
mod expression;

pub use error::{ParseError, ParseResult};

use crate::ast::node::*;
use crate::lexer::{Lexer, Token, TokenKind};
use crate::ast::node::{Precedence, Block, Statement};

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    next: Token,
    had_error: bool,
    typevar_scopes: Vec<std::collections::HashSet<String>>,
}

impl<'a> Parser<'a> {
    pub fn new(mut lexer: Lexer<'a>) -> ParseResult<Self> {
        let current = lexer.next_token().map_err(|e| ParseError::LexError(e))?;
        let next = lexer.next_token().map_err(|e| ParseError::LexError(e))?;
        
        Ok(Parser {
            lexer,
            current,
            next,
            had_error: false,
            typevar_scopes: vec![std::collections::HashSet::new()],
        })
    }

    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let mut items = Vec::new();
        while !self.is_at_end() {
            let attributes = self.parse_attributes()?;
            match self.parse_item_with_attributes(attributes.clone()) {
                Ok(item) => items.push(item),
                Err(e) => {
                    self.had_error = true;
                    eprintln!("Parse error: {}", e);
                    self.synchronize();
                }
            }
        }
        if self.had_error {
            return Err(ParseError::MultipleErrors);
        }
        Ok(Program { items })
    }

    fn parse_item_with_attributes(&mut self, attributes: Vec<Attribute>) -> ParseResult<Item> {
        match self.current.kind {
            TokenKind::Fn | TokenKind::Async => {
                let function = self.parse_function(attributes.clone())?;
                Ok(Item::Function(function))
            }
            TokenKind::Struct => {
                let structure = self.parse_struct(attributes.clone())?;
                Ok(Item::Struct(structure))
            }
            TokenKind::Enum => {
                let en = self.parse_enum(attributes.clone())?;
                Ok(Item::Enum(en))
            }
            TokenKind::Trait => {
                let trait_def = self.parse_trait(attributes.clone())?;
                Ok(Item::Trait(trait_def))
            }
            TokenKind::Impl => {
                let impl_block = self.parse_impl(attributes.clone())?;
                Ok(Item::Impl(impl_block))
            }
            TokenKind::Mod => {
                let module = self.parse_module()?;
                Ok(Item::Module(module))
            }
            TokenKind::Use | TokenKind::Import => {
                let imp = self.parse_import()?;
                Ok(Item::Import(imp))
            }
            TokenKind::Type => {
                let ta = self.parse_type_alias()?;
                Ok(Item::TypeAlias(ta))
            }
            TokenKind::Extern => {
                let ext = self.parse_extern(attributes.clone())?;
                Ok(Item::Extern(ext))
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: TokenKind::Fn,
                found: self.current.kind,
                line: self.current.line,
                column: self.current.column,
            }),
        }
    }


    fn parse_type_params(&mut self) -> ParseResult<Vec<TypeParameter>> {
        if !self.check(TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.advance()?;
        
        let mut params = Vec::new();
        
        while !self.check(TokenKind::Gt) && !self.is_at_end() {
            let tok = self.expect(TokenKind::Identifier)?;
            let span = Span::new(tok.line, tok.column, 0);
            let name = Identifier::new(tok.text.unwrap_or_default(), span);
            
            let mut bounds = Vec::new();
            
            if self.match_any(&[TokenKind::Colon]) {
                loop {
                    bounds.push(self.parse_type()?);
                    
                    if self.match_any(&[TokenKind::Plus]) {
                        continue;
                    } else {
                        break;
                    }
                }
            }
            
            params.push(TypeParameter {
                name,
                bounds,
                span,
            });
            
            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }
        
        self.expect_gt_or_shr()?;
        Ok(params)
    }




    fn parse_module(&mut self) -> ParseResult<ModuleDef> {
        let start_span = Span::new(self.current.line, self.current.column, 3);
        self.expect(TokenKind::Mod)?;

        let name_tok = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(
            name_tok.text.unwrap_or_default(),
            Span::new(name_tok.line, name_tok.column, 0),
        );

        self.expect(TokenKind::LBrace)?;

        let mut items = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let attrs = self.parse_attributes()?;
            match self.parse_item_with_attributes(attrs) {
                Ok(item) => items.push(item),
                Err(e) => {
                    self.had_error = true;
                    eprintln!("Parse error in module {}: {:?}", name.name, e);
                    self.synchronize();
                }
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(ModuleDef { name, items, span: start_span })
    }

    fn parse_extern(&mut self, attributes: Vec<Attribute>) -> ParseResult<ExternBlock> {
        let start_span = Span::new(self.current.line, self.current.column, 6);
        self.expect(TokenKind::Extern)?;
        
        let abi = if self.check(TokenKind::StringLiteral) {
            let tok = self.expect(TokenKind::StringLiteral)?;
            tok.text.unwrap_or_else(|| "C".to_string())
        } else {
            "C".to_string()
        };
        
        self.expect(TokenKind::LBrace)?;
        
        let mut functions = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let fn_attrs = self.parse_attributes()?;
            functions.push(self.parse_extern_fn(fn_attrs)?);
        }
        
        self.expect(TokenKind::RBrace)?;
        
        Ok(ExternBlock {
            abi,
            functions,
            span: start_span,
            attributes,
        })
    }

    fn parse_extern_fn(&mut self, attributes: Vec<Attribute>) -> ParseResult<ExternFn> {
        let start_span = Span::new(self.current.line, self.current.column, 2);
        
        let is_async = self.match_any(&[TokenKind::Async]);
        self.expect(TokenKind::Fn)?;
        
        let name_tok = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(
            name_tok.text.unwrap_or_default(),
            Span::new(name_tok.line, name_tok.column, 0),
        );
        
        self.expect(TokenKind::LParen)?;
        let params = self.parse_parameter_list()?;
        self.expect(TokenKind::RParen)?;
        
        let return_type = if self.match_any(&[TokenKind::Arrow]) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        self.expect(TokenKind::Semicolon)?;
        
        Ok(ExternFn {
            name,
            params,
            return_type,
            is_async,
            span: start_span,
            attributes,
        })
    }


    fn parse_import(&mut self) -> ParseResult<ImportStmt> {
        let start_span = Span::new(self.current.line, self.current.column, 0);

        match self.current.kind {
            TokenKind::Use | TokenKind::Import => {
                self.advance()?;
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: TokenKind::Use,
                    found: self.current.kind.clone(),
                    line: self.current.line,
                    column: self.current.column,
                });
            }
        }

        let mut segments = Vec::new();
        let first = self.expect(TokenKind::Identifier)?;
        segments.push(PathSeg::Ident(Identifier::new(
            first.text.unwrap_or_default(),
            Span::new(first.line, first.column, 0),
        )));
        while self.match_any(&[TokenKind::ColonColon]) {
            let seg = self.expect(TokenKind::Identifier)?;
            segments.push(PathSeg::Ident(Identifier::new(
                seg.text.unwrap_or_default(),
                Span::new(seg.line, seg.column, 0),
            )));
        }

        let path = Path {
            segments,
            span: start_span,
            resolved: None,
        };

        let alias = if self.match_any(&[TokenKind::As]) {
            let a = self.expect(TokenKind::Identifier)?;
            Some(Identifier::new(
                a.text.unwrap_or_default(),
                Span::new(a.line, a.column, 0),
            ))
        } else {
            None
        };

        self.expect(TokenKind::Semicolon)?;
        Ok(ImportStmt { path, alias, span: start_span })
    }






    pub fn current_token(&self) -> &Token {
        &self.current
    }

    pub fn peek_token(&self) -> &Token {
        &self.next
    }

    pub fn advance(&mut self) -> ParseResult<()> {
        std::mem::swap(&mut self.current, &mut self.next);
        self.next = self.lexer.next_token().map_err(|e| ParseError::LexError(e))?;
        Ok(())
    }

    fn is_at_end(&self) -> bool {
        self.current.kind == TokenKind::Eof
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.current.kind == kind
    }

    pub fn expect(&mut self, kind: TokenKind) -> ParseResult<Token> {
        if self.check(kind) {
            let token = self.current.clone();
            self.advance()?;
            Ok(token)
        } else {
            Err(ParseError::UnexpectedToken {
                expected: kind,
                found: self.current.kind.clone(),
                line: self.current.line,
                column: self.current.column,
            })
        }
    }

    fn match_any(&mut self, kinds: &[TokenKind]) -> bool {
        for kind in kinds {
            if self.check(*kind) {
                let _ = self.advance();
                return true;
            }
        }
        false
    }

    fn synchronize(&mut self) {
        let _ = self.advance();
        
        while !self.is_at_end() {
            if matches!(
                self.current.kind,
                TokenKind::Fn
                    | TokenKind::Let
                    | TokenKind::Return
                    | TokenKind::If
                    | TokenKind::While
                    | TokenKind::For
                    | TokenKind::Struct
                    | TokenKind::Enum
                    | TokenKind::Impl
            ) {
                return;
            }
            
            let _ = self.advance();
        }
    }

    fn parse_item(&mut self) -> ParseResult<Item> {

        let attributes = self.parse_attributes()?;

        match self.current.kind {
            TokenKind::Fn | TokenKind::Async => {

                self.parse_function(attributes.clone()).map(Item::Function)
            },
            TokenKind::Struct => self.parse_struct(attributes.clone()).map(Item::Struct),
            TokenKind::Enum => self.parse_enum(attributes.clone()).map(Item::Enum),
            TokenKind::Type => self.parse_type_alias().map(Item::TypeAlias),
            TokenKind::Trait => self.parse_trait(attributes.clone()).map(Item::Trait),
            TokenKind::Impl => self.parse_impl(attributes.clone()).map(Item::Impl),
            TokenKind::Mod => self.parse_module().map(Item::Module),
            TokenKind::Use | TokenKind::Import => self.parse_import().map(Item::Import),
            TokenKind::Extern => self.parse_extern(attributes.clone()).map(Item::Extern),
            _ => {

                Err(ParseError::UnexpectedToken {
                expected: TokenKind::Fn,
                found: self.current.kind.clone(),
                line: self.current.line,
                column: self.current.column,
            })
            },
        }
    }

    fn parse_function(&mut self, attributes: Vec<Attribute>) -> ParseResult<FunctionDef> {
        let start_span = Span::new(self.current.line, self.current.column, 2);
        
        let is_async = if self.match_any(&[TokenKind::Async]) {
            true
        } else {
            false
        };

        self.expect(TokenKind::Fn)?;
        
        let name_token = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(
            name_token.text.unwrap_or_default(),
            Span::new(name_token.line, name_token.column, 0),
        );
        
        let type_params = if self.check(TokenKind::Lt) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };
        
        self.expect(TokenKind::LParen)?;
        let params = self.parse_parameter_list()?;
        self.expect(TokenKind::RParen)?;

        let return_type = if self.match_any(&[TokenKind::Arrow]) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let body = self.parse_block()?;
        
        Ok(FunctionDef {
            name,
            type_params,
            params,
            return_type,
            body,
            is_async,
            span: start_span,
            attributes,
            defid: None, 
        })
    }

    fn parse_parameter_list(&mut self) -> ParseResult<Vec<Parameter>> {
        let mut params = Vec::new();
        
        if self.check(TokenKind::RParen) {
            return Ok(params);
        }
        
        loop {
            let name_token = match self.current.kind {
                TokenKind::Identifier | TokenKind::SelfLower => {
                    let tok = self.current.clone();
                    self.advance()?;
                    tok
                },
                _ => return Err(ParseError::UnexpectedToken {
                    expected: TokenKind::Identifier,
                    found: self.current.kind.clone(),
                    line: self.current.line,
                    column: self.current.column,
                }),
            };

            let name_str = name_token.text().unwrap_or("");
            let span = Span::new(name_token.line, name_token.column, 0);

            let ty = if name_str == "self" {
                if self.match_any(&[TokenKind::Colon]) {
                    self.parse_type()?
                } else {
                    Type::Named { 
                        name: Self::simple_ident_path("Self", span.clone()),
                        type_args: vec![] 
                    }
                }
            } else {
                self.expect(TokenKind::Colon)?;
                self.parse_type()?
            };

            let name = Identifier::new(
                name_str.to_string(),
                span.clone(),
            );

            params.push(Parameter {
                name,
                ty,
                span,
            });

            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }

        Ok(params)
    }

    fn parse_struct(&mut self, attributes: Vec<Attribute>) -> ParseResult<StructDef> {
        let startspan = Span::new(self.current.line, self.current.column, 6);
        self.expect(TokenKind::Struct)?;
        
        let nametoken = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(
            nametoken.text.unwrap_or_default(),
            Span::new(nametoken.line, nametoken.column, 0),
        );

        let type_params = self.parse_type_params()?;
        
        self.push_typevars(&type_params);
        
        self.expect(TokenKind::LBrace)?;
        let fields = self.parse_field_list()?;
        self.expect(TokenKind::RBrace)?;
        
        self.pop_typevars();

        Ok(StructDef {
            name,
            fields,
            type_params,
            where_clause: None,
            span: startspan,
            attributes,
        })
    }





    fn parse_attributes(&mut self) -> ParseResult<Vec<Attribute>> {
        let mut attrs = Vec::new();
        while self.check(TokenKind::Pound) {
            attrs.push(self.parse_attribute()?);
        }
        Ok(attrs)
    }

    fn parse_attribute(&mut self) -> ParseResult<Attribute> {
        self.expect(TokenKind::Pound)?;
        self.expect(TokenKind::LBracket)?;
        
        let attr = self.parse_attribute_inner()?;
        
        self.expect(TokenKind::RBracket)?;
        Ok(attr)
    }

    fn parse_attribute_inner(&mut self) -> ParseResult<Attribute> {
        let name_str = if self.current.kind == TokenKind::Identifier || self.current.kind.is_reserved_keyword() {
            let s = self.current.text().unwrap_or_default().to_string();
            self.advance()?;
            s
        } else {
            return Err(ParseError::UnexpectedToken {
                expected: TokenKind::Identifier,
                found: self.current.kind.clone(),
                line: self.current.line,
                column: self.current.column,
            });
        };
        let id = Identifier::new(
            name_str,
            Span::new(self.current.line, self.current.column, 0),
        );
        
        if id.name == "derive" && self.check(TokenKind::LParen) {
            self.advance()?;
            let mut traits = Vec::new();
            loop {
                let trait_token = self.expect(TokenKind::Identifier)?;
                traits.push(trait_token.text().unwrap_or("").to_string());
                if !self.match_any(&[TokenKind::Comma]) {
                    break;
                }
            }
            self.expect(TokenKind::RParen)?;
            return Ok(Attribute::Derive(traits));
        }

        if self.match_any(&[TokenKind::Assign]) {
            let val = if self.check(TokenKind::StringLiteral) {
                let tok = self.advance_clone()?;
                Literal::String(tok.text.unwrap_or_default())
            } else if self.check(TokenKind::IntLiteral) {
                let tok = self.advance_clone()?;
                Literal::Int(tok.text.unwrap_or_default().parse().unwrap_or(0))
            } else if self.check(TokenKind::FloatLiteral) {
                let tok = self.advance_clone()?;
                Literal::Float(tok.text.unwrap_or_default().parse().unwrap_or(0.0))
            } else if self.match_any(&[TokenKind::True]) {
                Literal::Bool(true)
            } else if self.match_any(&[TokenKind::False]) {
                Literal::Bool(false)
            } else {
                return Err(ParseError::UnexpectedToken {
                    expected: TokenKind::StringLiteral,
                    found: self.current.kind,
                    line: self.current.line,
                    column: self.current.column,
                });
            };
            Ok(Attribute::Value(id, val))
        } else if self.match_any(&[TokenKind::LParen]) {
            let mut inner = Vec::new();
            while !self.check(TokenKind::RParen) && !self.is_at_end() {
                inner.push(self.parse_attribute_inner()?);
                if !self.match_any(&[TokenKind::Comma]) {
                    break;
                }
            }
            self.expect(TokenKind::RParen)?;
            Ok(Attribute::Nested(id, inner))
        } else {
            Ok(Attribute::Simple(id))
        }
    }


    fn advance_clone(&mut self) -> ParseResult<Token> {
        let tok = self.current.clone();
        self.advance()?;
        Ok(tok)
    }

    fn parse_field_list(&mut self) -> ParseResult<Vec<StructField>> {
        let mut fields = Vec::new();
        
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let name_token = self.expect(TokenKind::Identifier)?;
            
            let name = Identifier::new(
                name_token.text().unwrap_or("").to_string(),
                Span::new(name_token.line, name_token.column, 0),
            );

            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;

            fields.push(StructField {
                name,
                ty,
                span: Span::new(name_token.line, name_token.column, 0),
            });


            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }
        
        Ok(fields)
    }

    fn parse_enum(&mut self, attributes: Vec<Attribute>) -> ParseResult<EnumDef> {
        let startspan = Span::new(self.current.line, self.current.column, 4);
        self.expect(TokenKind::Enum)?;

        let nametoken = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(
            nametoken.text.clone().unwrap_or_default(),
            Span::new(nametoken.line, nametoken.column, 0),
        );


        let type_params = self.parse_type_params()?;
        self.push_typevars(&type_params);

        self.expect(TokenKind::LBrace)?;

        let mut variants = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let vtok = self.expect(TokenKind::Identifier)?;
            let vname = Identifier::new(
                vtok.text.clone().unwrap_or_default(),
                Span::new(vtok.line, vtok.column, 0),
            );

            let data = if self.check(TokenKind::LParen) {
                self.expect(TokenKind::LParen)?;
                let tys = self.parse_type_list()?;
                self.expect(TokenKind::RParen)?;
                VariantData::Tuple(tys)
            } else if self.check(TokenKind::LBrace) {
                self.expect(TokenKind::LBrace)?;
                let fields = self.parse_field_list()?;
                self.expect(TokenKind::RBrace)?;
                VariantData::Struct(fields)
            } else {
                VariantData::Unit
            };

            let span = Span::new(vtok.line, vtok.column, 0);
            variants.push(EnumVariant { name: vname, data, span });

            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }

        self.expect(TokenKind::RBrace)?;

        self.pop_typevars();

        Ok(EnumDef {
            name,
            variants,
            type_params,
            span: startspan,
            attributes,
        })
    }

    fn parse_type_alias(&mut self) -> ParseResult<TypeAlias> {
        let startspan = Span::new(self.current.line, self.current.column, 4);
        self.expect(TokenKind::Type)?;

        let nametoken = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(
            nametoken.text.clone().unwrap_or_default(),
            Span::new(nametoken.line, nametoken.column, 0),
        );

        let type_params = self.parse_type_params()?; 

        self.expect(TokenKind::Assign)?;
        self.push_typevars(&type_params);
        let ty = self.parse_type()?;
        self.pop_typevars(); 
        self.expect(TokenKind::Semicolon)?;

        Ok(TypeAlias {
            name,
            type_params,
            ty,
            span: startspan,
        })
    }

    fn parse_trait(&mut self, _attributes: Vec<Attribute>) -> ParseResult<TraitDef> {
        let start_span = Span::new(self.current.line, self.current.column, 5);
        self.expect(TokenKind::Trait)?;
        let name_token = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(
            name_token.text().unwrap_or(""),
            Span::new(name_token.line, name_token.column, 0),
        );

        let type_params = self.parse_type_params()?;

        self.push_typevars(&type_params);

        self.expect(TokenKind::LBrace)?;

        let mut methods = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            methods.push(self.parse_trait_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        self.pop_typevars();

        Ok(TraitDef {
            name: name.clone(),
            type_params,
            assoc_types: vec![], 
            methods,
            span: name.span,
        })
    }

    fn parse_trait_item(&mut self) -> ParseResult<TraitMethod> {
        if self.match_any(&[TokenKind::Type]) {
            let _name = self.expect(TokenKind::Identifier)?;
            self.expect(TokenKind::Semicolon)?;
            return Ok(TraitMethod {
                name: Identifier::new("DUMMY_ASSOCIATED_TYPE", Span::initial()),
                params: vec![],
                return_type: None,
                span: Span::initial(),
                defid: None,
            });
        }
        
        self.expect(TokenKind::Fn)?;
        let name_tok = self.expect(TokenKind::Identifier)?;
        let name = Identifier::new(name_tok.text.unwrap_or_default(), Span::new(name_tok.line, name_tok.column, 0));
        self.expect(TokenKind::LParen)?;
        let params = self.parse_parameter_list()?;
        self.expect(TokenKind::RParen)?;
        let return_type = if self.match_any(&[TokenKind::Arrow]) { Some(self.parse_type()?) } else { None };
        self.expect(TokenKind::Semicolon)?;
        Ok(TraitMethod { 
            name, 
            params, 
            return_type, 
            span: Span::initial(),
            defid: None, 
        })
    }

    fn parse_impl(&mut self, attributes: Vec<Attribute>) -> ParseResult<ImplBlock> {
        let start_span = Span::new(self.current.line, self.current.column, 0);
        self.expect(TokenKind::Impl)?;
        
        let type_params = self.parse_type_params()?;

        self.push_typevars(&type_params);

        let first_ty = self.parse_type()?;
        
        let (trait_ref, self_ty) = if self.match_any(&[TokenKind::For]) {
             let t_ref = Some(first_ty);
             let s_ty = self.parse_type()?;
             (t_ref, s_ty)
        } else {
             (None, first_ty)
        };

        let mut where_preds = Vec::new();
        if self.current.kind == TokenKind::Identifier && self.current.text.as_deref() == Some("where") {
            self.advance()?; 

            loop {
                if self.check(TokenKind::LBrace) { break; }

                let target_ty = self.parse_type()?;
                self.expect(TokenKind::Colon)?;
                let bound_ty = self.parse_type()?;

                where_preds.push(crate::ast::node::WherePredicate::Bound {
                    target_ty,
                    bound_ty,
                    span: self.current.span(),
                });

                if !self.match_any(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        
        if self.current.kind == TokenKind::Identifier && self.current.text.as_deref() == Some("where") {
             self.advance()?; 
             while !self.check(TokenKind::LBrace) && !self.is_at_end() {
                 self.advance()?;
             }
        }

        self.expect(TokenKind::LBrace)?;

        let mut items = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            items.push(self.parse_impl_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        self.pop_typevars();

        let span = Span::new(start_span.line, start_span.column, 0); 

        Ok(ImplBlock {
            type_params,
            trait_ref: trait_ref,
            self_ty: self_ty,
            items,
            span,
            where_preds, 
        })
    }

    fn parse_impl_item(&mut self) -> ParseResult<ImplItem> {
        let attributes = self.parse_attributes()?;

        if self.match_any(&[TokenKind::Type]) {
            let name_token = self.expect(TokenKind::Identifier)?;
            let name = Identifier::new(
                name_token.text.unwrap_or_default(),
                Span::new(name_token.line, name_token.column, 0)
            );
            
            self.expect(TokenKind::Assign)?; 
            let ty = self.parse_type()?;
            self.expect(TokenKind::Semicolon)?;

            return Ok(ImplItem::AssocType(AssocTypeBinding {
                name,
                ty,
                span: Span::new(name_token.line, name_token.column, 0),
            }));
        }

        let func = self.parse_function(attributes)?;
        Ok(ImplItem::Method(func))
    }

    pub fn parse_type(&mut self) -> ParseResult<Type> {
        match self.current.kind {
            TokenKind::Identifier | TokenKind::SelfLower | TokenKind::SelfUpper => {
                let start_span = Span::new(self.current.line, self.current.column, 0);
                
                let mut segments = Vec::new();
                loop {
                    let segment = if self.match_any(&[TokenKind::SelfLower]) {
                         PathSeg::Ident(Identifier::new("Self", start_span)) 
                    } else if self.match_any(&[TokenKind::SelfUpper]) {
                         PathSeg::Ident(Identifier::new("Self", start_span))
                    } else {
                         let tok = self.expect(TokenKind::Identifier)?;
                         PathSeg::Ident(Identifier::new(tok.text.unwrap_or_default(), Span::new(tok.line, tok.column, 0)))
                    };
                    segments.push(segment);
                    
                    if self.check(TokenKind::ColonColon) {
                        self.advance()?;
                        continue;
                    }
                    break;
                }
                
                let path = Path { segments, span: start_span, resolved: None };


                if path.segments.len() == 1 {
                    if let PathSeg::Ident(id) = &path.segments[0] {
                        if id.name == "dyn" {
                            let trait_start = Span::new(self.current.line, self.current.column, 0);
                            let trait_path = self.parse_ident_path_only(trait_start)?;

                            if matches!(self.current.kind, TokenKind::Lt | TokenKind::LBracket) {
                                return Err(ParseError::InvalidSyntax {
                                    message: "generic dyn trait is not supported yet".to_string(),
                                    line: self.current.line,
                                    column: self.current.column,
                                });
                            }

                            if self.is_reserved_type_name_path(&trait_path) {
                                return Err(ParseError::InvalidSyntax {
                                    message: "dyn must be followed by a trait path (e.g., dyn foo::Bar)".to_string(),
                                    line: self.current.line,
                                    column: self.current.column,
                                });
                            }

                            return Ok(Type::DynTrait { trait_path });
                        }
                    }
                }

                if path.segments.len() == 1 {
                    if let PathSeg::Ident(id) = &path.segments[0] {
                        if self.is_typevar(id.name.as_str()) {
                            return Ok(Type::TypeVar(id.clone()));
                        }
                    }
                }

                if path.segments.len() == 1 {
                    if let PathSeg::Ident(id) = &path.segments[0] {
                        match id.name.as_str() {
                            "Int" => return Ok(Type::Int),
                            "Float" => return Ok(Type::Float),
                            "Bool" => return Ok(Type::Bool),
                            "String" => return Ok(Type::String),
                            "Unit" => return Ok(Type::Unit),
                            "Gaussian" => return Ok(Type::Gaussian),
                            "Uniform" => return Ok(Type::Uniform),
                            "Any" => return Ok(Type::Any),
                            "Map" => {
                                if self.match_any(&[TokenKind::Lt]) {
                                    let key = self.parse_type()?;
                                    self.expect(TokenKind::Comma)?;
                                    let value = self.parse_type()?;
                                    self.expect_gt_or_shr()?;
                                    return Ok(Type::Map(Box::new(key), Box::new(value)));
                                }
                            },
                            "Option" => {
                                if self.check(TokenKind::Lt) {
                                    self.advance()?;
                                    let inner = self.parse_type()?;
                                    self.expect_gt_or_shr()?;
                                    return Ok(Type::Option(Box::new(inner)));
                                } else if self.check(TokenKind::LBracket) {
                                    self.advance()?;
                                    let inner = self.parse_type()?;
                                    self.expect(TokenKind::RBracket)?;
                                    return Ok(Type::Option(Box::new(inner)));
                                }
                            }
                            "Result" => {
                                let is_bracket = if self.check(TokenKind::Lt) {
                                    self.advance()?;
                                    false
                                } else if self.check(TokenKind::LBracket) {
                                    self.advance()?;
                                    true
                                } else {
                                    false 
                                };

                                if self.current.kind != TokenKind::Identifier && !is_bracket {
                                } else {
                                     let ok_type = self.parse_type()?;
                                     self.expect(TokenKind::Comma)?;
                                     let err_type = self.parse_type()?;
                                     if is_bracket {
                                         self.expect(TokenKind::RBracket)?;
                                     } else {
                                         self.expect_gt_or_shr()?;
                                     }
                                     return Ok(Type::Result {
                                         ok_type: Box::new(ok_type),
                                         err_type: Box::new(err_type),
                                     });
                                }
                            }
                            "Rc" => {
                                 if self.match_any(&[TokenKind::Lt]) {
                                     let inner = self.parse_type()?;
                                     self.expect_gt_or_shr()?;
                                     return Ok(Type::Rc(Box::new(inner)));
                                 }
                            }
                            "Weak" => {
                                if self.match_any(&[TokenKind::Lt]) {
                                    let inner = self.parse_type()?;
                                    self.expect_gt_or_shr()?;
                                    return Ok(Type::Weak(Box::new(inner)));
                                }
                            }
                            "Handle" => {
                                if self.match_any(&[TokenKind::Lt]) {
                                    let inner = self.parse_type()?;
                                    self.expect_gt_or_shr()?;
                                    return Ok(Type::Handle(Box::new(inner)));
                                }
                            }
                            "Signal" => {
                                if self.match_any(&[TokenKind::Lt]) {
                                    let inner = self.parse_type()?;
                                    self.expect_gt_or_shr()?;
                                    return Ok(Type::Signal(Box::new(inner)));
                                }
                            }
                            "Event" => {
                                if self.match_any(&[TokenKind::Lt]) {
                                    let inner = self.parse_type()?;
                                    self.expect_gt_or_shr()?;
                                    return Ok(Type::Event(Box::new(inner)));
                                }
                            }
                            "Array" => {
                                if self.check(TokenKind::Lt) {
                                    self.advance()?;
                                    let inner = self.parse_type()?;
                                    self.expect_gt_or_shr()?;
                                    return Ok(Type::Array(Box::new(inner)));
                                } else if self.check(TokenKind::LBracket) {
                                    self.advance()?;
                                    let inner = self.parse_type()?;
                                    self.expect(TokenKind::RBracket)?;
                                    return Ok(Type::Array(Box::new(inner)));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if path.segments.len() == 2 {
                    if let (PathSeg::Ident(lhs), PathSeg::Ident(rhs)) = (&path.segments[0], &path.segments[1]) {
                        let is_self = lhs.name == "Self";
                        let is_tvar = self.is_typevar(&lhs.name);

                        if is_self || is_tvar {
                            let selfty = if is_tvar {
                                Type::TypeVar(lhs.clone())
                            } else {
                                Type::Named { name: Path::from_ident(lhs.clone()), type_args: vec![] }
                            };

                            return Ok(Type::Assoc {
                                trait_def: None,
                                self_ty: Box::new(selfty),
                                name: rhs.name.clone(),
                            });
                        }
                    }
                }
                
                let type_args = if self.check(TokenKind::Lt) || self.check(TokenKind::LBracket) {
                    self.parse_type_arguments()?
                } else {
                    Vec::new()
                };
                
                Ok(Type::Named { name: path, type_args })
            }
            TokenKind::Fn => {
                self.advance()?;
                self.expect(TokenKind::LParen)?;
                let params = self.parse_type_list()?;
                self.expect(TokenKind::RParen)?;
                self.expect(TokenKind::Arrow)?;
                let return_type = Box::new(self.parse_type()?);
                Ok(Type::Function { params, return_type })
            }
            
            TokenKind::LParen => {
                self.advance()?;
                if self.current.kind == TokenKind::RParen {
                    self.advance()?;
                    return Ok(Type::Unit);
                }
                let types = self.parse_type_list()?;
                self.expect(TokenKind::RParen)?;
                if types.len() == 1 {
                    Ok(types.into_iter().next().unwrap())
                } else {
                    Ok(Type::Tuple(types))
                }
            }
            
            TokenKind::LBracket => {
                self.advance()?;
                let inner = self.parse_type()?;
                self.expect(TokenKind::RBracket)?;
                Ok(Type::Array(Box::new(inner)))
            }
            _ => {
                 if self.check(TokenKind::LBracket) {
                     self.advance()?;
                     let inner = self.parse_type()?;
                     self.expect(TokenKind::RBracket)?;
                     return Ok(Type::Array(Box::new(inner)));
                 }
                 Err(ParseError::UnexpectedToken {
                    expected: TokenKind::Identifier, 
                    found: self.current.kind.clone(),
                    line: self.current.line,
                    column: self.current.column,
                 })
            }
        }
    }

    fn parse_ident_path_only(&mut self, startspan: Span) -> ParseResult<Path> {
        let mut segments = Vec::new();

        let first = self.expect(TokenKind::Identifier)?;
        segments.push(PathSeg::Ident(Identifier::new(
            first.text.unwrap_or_default(),
            Span::new(first.line, first.column, 0),
        )));

        while self.match_any(&[TokenKind::ColonColon]) {
            let seg = self.expect(TokenKind::Identifier)?;
            segments.push(PathSeg::Ident(Identifier::new(
                seg.text.unwrap_or_default(),
                Span::new(seg.line, seg.column, 0),
            )));
        }

        Ok(Path { segments, span: startspan, resolved: None })
    }

    fn is_reserved_type_name_path(&self, p: &Path) -> bool {
        if p.segments.len() != 1 { return false; }
        let PathSeg::Ident(id) = &p.segments[0] else { return false; };
        matches!(id.name.as_str(),
            "Int" | "Float" | "Bool" | "String" | "Unit" |
            "Gaussian" | "Uniform" |
            "Option" | "Result" | "Array" |
            "Rc" | "Weak" |
            "Signal" | "Event"
        )
    }

    fn expect_gt_or_shr(&mut self) -> ParseResult<()> {
        match self.current.kind {
            TokenKind::Gt => {
                self.advance()?;
                Ok(())
            }
            TokenKind::Shr => {
                self.current.kind = TokenKind::Gt;
                Ok(())
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: TokenKind::Gt,
                found: self.current.kind.clone(),
                line: self.current.line,
                column: self.current.column,
            }),
        }
    }

    fn push_typevars(&mut self, params: &[TypeParameter]) {
        let mut set = std::collections::HashSet::new();
        for p in params {
            set.insert(p.name.name.clone());
        }
        self.typevar_scopes.push(set);
    }

    fn pop_typevars(&mut self) {
        self.typevar_scopes.pop();
    }

    fn is_typevar(&self, s: &str) -> bool {
        self.typevar_scopes.iter().rev().any(|set| set.contains(s))
    }

    fn simple_ident_path(name: &str, span: Span) -> Path {
        Path {
            segments: vec![PathSeg::Ident(Identifier::new(name.to_string(), span))],
            span,
            resolved: None,
        }
    }

    fn parse_type_list(&mut self) -> ParseResult<Vec<Type>> {
        let mut types = Vec::new();
        
        if self.check(TokenKind::RParen) {
            return Ok(types);
        }
        
        loop {
            types.push(self.parse_type()?);
            
            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }
        
        Ok(types)
    }

    fn parse_type_arguments(&mut self) -> ParseResult<Vec<Type>> {
        if !self.check(TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.advance()?;

        let mut args = Vec::new();
        loop {
            args.push(self.parse_type()?);

            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }

        self.expect_gt_or_shr()?;
        Ok(args)
    }

    pub fn parse_statement(&mut self) -> ParseResult<Statement> {
        
        match self.current.kind {
            TokenKind::Let => self.parse_let_statement(),
            TokenKind::Return => self.parse_return_statement(),
            TokenKind::While => self.parse_while_statement(),
            TokenKind::For => self.parse_for_statement(),
            TokenKind::Break => self.parse_break_statement(),
            TokenKind::Continue => self.parse_continue_statement(),
            TokenKind::Semicolon => {
                self.advance()?;
                Ok(Statement::Empty)
            }
            TokenKind::With => {
                let expr = self.parse_expression()?;
                if self.match_any(&[TokenKind::Semicolon]) {
                }
                Ok(Statement::Expression(expr))
            }
            _ => {
                let expr = self.parse_expression()?;
                if self.match_any(&[TokenKind::Semicolon]) {
                }
                Ok(Statement::Expression(expr))
            }
        }
    }

    fn parse_let_statement(&mut self) -> ParseResult<Statement> {
        let start_span = Span::new(self.current.line, self.current.column, 3);
        self.expect(TokenKind::Let)?;
        
        if self.current.kind == TokenKind::Identifier && self.current.text().unwrap_or("") == "mut" {
            self.advance()?;
        } else if self.match_any(&[
            //TokenKind::Mut, 
        ]) {

        }

        let pattern = self.parse_pattern()?;
        
        let ty = if self.match_any(&[TokenKind::Colon]) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let init = if self.match_any(&[TokenKind::Assign]) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.expect(TokenKind::Semicolon)?;
        
        Ok(Statement::Let {
            pattern,
            ty,
            init,
            span: start_span,
        })
    }

    fn parse_return_statement(&mut self) -> ParseResult<Statement> {
        let start_span = Span::new(self.current.line, self.current.column, 6);
        self.expect(TokenKind::Return)?;
        
        let value = if !self.check(TokenKind::Semicolon) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        self.expect(TokenKind::Semicolon)?;
        
        Ok(Statement::Return {
            value,
            span: start_span,
        })
    }

    fn parse_while_statement(&mut self) -> ParseResult<Statement> {
        let span = Span::new(self.current.line, self.current.column, 5);
        self.expect(TokenKind::While)?;
        
        let condition = self.parse_expression()?;
        let body = self.parse_block()?;
        
        Ok(Statement::While {
            condition,
            body,
            span,
        })
    }

    fn parse_for_statement(&mut self) -> ParseResult<Statement> {
        let span = Span::new(self.current.line, self.current.column, 3);
        self.expect(TokenKind::For)?;
        
        let pattern = self.parse_pattern()?;
        
        let in_token = self.expect(TokenKind::Identifier)?;
        if in_token.text() != Some("in") {
            return Err(ParseError::UnexpectedToken {
                expected: TokenKind::Identifier,
                found: self.current.kind,
                line: self.current.line,
                column: self.current.column,
            });
        }
        
        let iterator = self.parse_expression()?;
        let body = self.parse_block()?;
        
        Ok(Statement::For {
            pattern,
            iterator,
            body,
            span,
        })
    }

    fn parse_break_statement(&mut self) -> ParseResult<Statement> {
        let span = Span::new(self.current.line, self.current.column, 5);
        self.expect(TokenKind::Break)?;
        self.expect(TokenKind::Semicolon)?;
        Ok(Statement::Break { span })
    }

    fn parse_continue_statement(&mut self) -> ParseResult<Statement> {
        let span = Span::new(self.current.line, self.current.column, 8);
        self.expect(TokenKind::Continue)?;
        self.expect(TokenKind::Semicolon)?;
        Ok(Statement::Continue { span })
    }

    fn parse_block(&mut self) -> ParseResult<Block> {
        let start_span = Span::new(self.current.line, self.current.column, 1);
        self.expect(TokenKind::LBrace)?;
        
        let mut statements = Vec::new();
        
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            statements.push(self.parse_statement()?);
        }
        
        self.expect(TokenKind::RBrace)?;
        
        Ok(Block {
            statements,
            span: start_span,
        })
    }

    pub fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_expression_with_precedence(Precedence::Lowest)
    }

    fn parse_primary_expression(&mut self) -> ParseResult<Expression> {
        match &self.current.kind {
            TokenKind::IntLiteral => self.parse_int_literal(),
            TokenKind::FloatLiteral => self.parse_float_literal(),
            TokenKind::StringLiteral => self.parse_string_literal(),
            TokenKind::True | TokenKind::False => self.parse_bool_literal(),
            TokenKind::Identifier | TokenKind::SelfLower => self.parse_identifier_or_call(),
            TokenKind::LParen => self.parse_paren_or_tuple(),
            TokenKind::If => self.parse_if_expression(),
            TokenKind::LBrace => self.parse_block_expression(),
            TokenKind::Minus | TokenKind::Not | TokenKind::BitNot => self.parse_unary_expression(),
            TokenKind::Match => self.parse_match_expression(),
            TokenKind::LBracket => self.parse_array_literal(),
            TokenKind::BitOr => self.parse_closure_expression(),
            _ => Err(ParseError::UnexpectedToken {
                expected: TokenKind::Identifier,
                found: self.current.kind.clone(),
                line: self.current.line,
                column: self.current.column,
            }),
        }
    }

    fn parse_array_literal(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 1);
        self.expect(TokenKind::LBracket)?;
        
        let mut elements = Vec::new();
        while !self.check(TokenKind::RBracket) && !self.is_at_end() {
            elements.push(self.parse_expression()?);
            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }
        
        self.expect(TokenKind::RBracket)?;
        Ok(Expression::Array { elements, span })
    }

    fn parse_closure_expression(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 1);
        self.expect(TokenKind::BitOr)?;
        
        let mut params = Vec::new();
        while !self.check(TokenKind::BitOr) && !self.is_at_end() {
            let name_token = self.expect(TokenKind::Identifier)?;
            let name = Identifier::new(
                name_token.text().unwrap_or("").to_string(), 
                Span::new(name_token.line, name_token.column, 0)
            );
            
            let ty = if self.match_any(&[TokenKind::Colon]) {
                self.parse_type()?
            } else {
                Type::Unit // 推論待ち
            };
            
            params.push(Parameter {
                name,
                ty,
                span: Span::new(name_token.line, name_token.column, 0),
            });

            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }
        
        self.expect(TokenKind::BitOr)?;
        
        let body = Box::new(self.parse_expression()?);
        
        Ok(Expression::Closure { params, body, span })
    }

    fn parse_int_literal(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 0);
        let text = self.current.text().unwrap_or("0");
        
        let value = if text.starts_with("0x") {
            i64::from_str_radix(&text[2..], 16)
        } else if text.starts_with("0o") {
            i64::from_str_radix(&text[2..], 8)
        } else if text.starts_with("0b") {
            i64::from_str_radix(&text[2..], 2)
        } else {
            text.parse()
        };
        
        let value = value.map_err(|_| ParseError::InvalidLiteral {
            literal: text.to_string(),
            reason: "Invalid number format".to_string(),
            line: self.current.line,
            column: self.current.column,
        })?;
        
        self.advance()?;
        
        Ok(Expression::Literal {
            value: Literal::Int(value),
            span,
        })
    }

    fn parse_float_literal(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 0);
        let text = self.current.text().unwrap_or("0.0");
        let value = text.parse().map_err(|_| ParseError::InvalidLiteral {
            literal: text.to_string(),
            reason: "Invalid number format".to_string(),
            line: self.current.line,
            column: self.current.column,
        })?;
        
        self.advance()?;
        
        Ok(Expression::Literal {
            value: Literal::Float(value),
            span,
        })
    }

    fn parse_string_literal(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 0);
        let text = self.current.text().unwrap_or("").to_string();
        
        self.advance()?;
        
        Ok(Expression::Literal {
            value: Literal::String(text),
            span,
        })
    }

    fn parse_bool_literal(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 0);
        let value = self.current.kind == TokenKind::True;
        
        self.advance()?;
        
        Ok(Expression::Literal {
            value: Literal::Bool(value),
            span,
        })
    }

    fn parse_identifier_or_call(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 0);

        let name_str = if self.current.kind == TokenKind::SelfLower {
            "self".to_string()
        } else {
            self.current.text.clone().unwrap_or_default().to_string()
        };

        let enum_name_ident = Identifier::new(name_str.clone(), span);

        self.advance()?;

        if name_str == "None" && self.current.kind != TokenKind::LParen {
            return Ok(Expression::None { span });
        }

        if self.current.kind == TokenKind::ColonColon {
            self.advance()?;

            let vtok = self.expect(TokenKind::Identifier)?;
            let vspan = Span::new(vtok.line, vtok.column, 0);
            let variant_ident = Identifier::new(vtok.text.unwrap_or_default(), vspan);
            let islower = variant_ident
                .name
                .chars()
                .next()
                .map(|c| c.is_lowercase())
                .unwrap_or(false);
            if islower {
                let traitpath = Path {
                    segments: vec![PathSeg::Ident(enum_name_ident.clone())],
                    span,
                    resolved: None,
                };

                let ufcs_expr = Expression::UfcsMethod {
                    trait_path: traitpath,
                    method: variant_ident,
                    span,
                };

                if self.check(TokenKind::LParen) {
                    self.advance()?;
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
                        span,
                    });
                }
                
                return Ok(ufcs_expr);
            }

            if self.check(TokenKind::LParen) {
                let mut args = Vec::new();
                self.advance()?;

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
                    name: enum_name_ident,
                    variant: variant_ident,
                    args,
                    named_fields: None,
                    span,
                });
            }
            else if self.check(TokenKind::LBrace) {
                self.advance()?;
                let mut args: Vec<Expression> = Vec::new();

                if !self.check(TokenKind::RBrace) {
                    loop {
                        self.expect(TokenKind::Identifier)?;
                        self.expect(TokenKind::Colon)?;
                        let value = self.parse_expression()?;
                        args.push(value);

                        if !self.match_any(&[TokenKind::Comma]) {
                            break;
                        }
                    }
                }

                self.expect(TokenKind::RBrace)?;

                return Ok(Expression::Enum {
                    name: enum_name_ident,
                    variant: variant_ident,
                    args,
                    named_fields: None,
                    span,
                });
            } else {
                let path = Path {
                    segments: vec![PathSeg::Ident(enum_name_ident), PathSeg::Ident(variant_ident)],
                    span,
                    resolved: None,
                };
                return Ok(Expression::Variable { name: path, span });
            }
        }

        let path = Path {
            segments: vec![PathSeg::Ident(enum_name_ident)],
            span,
            resolved: None,
        };
        Ok(Expression::Variable { name: path, span })
    }

    fn parse_argument_list(&mut self) -> ParseResult<Vec<Expression>> {
        let mut args = Vec::new();
        
        if self.check(TokenKind::RParen) {
            return Ok(args);
        }
        
        loop {
            args.push(self.parse_expression()?);
            
            if !self.match_any(&[TokenKind::Comma]) {
                break;
            }
        }
        
        Ok(args)
    }

    fn parse_paren_or_tuple(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 1);
        self.advance()?;
        
        if self.check(TokenKind::RParen) {
            self.advance()?;
            return Ok(Expression::Literal {
                value: Literal::Unit,
                span,
            });
        }
        
        let first_expr = self.parse_expression()?;
        
        if self.match_any(&[TokenKind::Comma]) {
            let mut elements = vec![first_expr];
            
            while !self.check(TokenKind::RParen) && !self.is_at_end() {
                elements.push(self.parse_expression()?);
                
                if !self.match_any(&[TokenKind::Comma]) {
                    break;
                }
            }
            
            self.expect(TokenKind::RParen)?;
            
            Ok(Expression::Tuple { elements, span })
        } else {
            self.expect(TokenKind::RParen)?;
            
            Ok(Expression::Paren {
                expr: Box::new(first_expr),
                span,
            })
        }
    }

    fn current_precedence(&self) -> Precedence {
        self.get_precedence(&self.current.kind)
    }

    fn get_precedence(&self, kind: &TokenKind) -> Precedence {
        match kind {
            TokenKind::Eq | TokenKind::Ne => Precedence::Equals,
            TokenKind::Lt | TokenKind::Gt | TokenKind::Le | TokenKind::Ge => Precedence::LessGreater,
            TokenKind::Plus | TokenKind::Minus => Precedence::Sum,
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Precedence::Product,
            TokenKind::LParen => Precedence::Call,
            TokenKind::Dot => Precedence::Call, 
            TokenKind::LBracket => Precedence::Index,
            TokenKind::Assign => Precedence::Equals, 
            TokenKind::PlusAssign | TokenKind::MinusAssign |
            TokenKind::StarAssign | TokenKind::SlashAssign |
            TokenKind::PercentAssign |
            TokenKind::BitAndAssign | TokenKind::BitOrAssign |
            TokenKind::BitXorAssign | TokenKind::ShlAssign |
            TokenKind::ShrAssign => Precedence::Equals,
            TokenKind::And => Precedence::LogicalAnd,
            TokenKind::Or => Precedence::LogicalOr,
            _ => Precedence::Lowest,
        }
    }
    
    fn current_binary_op(&self) -> Option<BinaryOp> {
        match self.current.kind {
            TokenKind::Plus => Some(BinaryOp::Add),
            TokenKind::Minus => Some(BinaryOp::Sub),
            TokenKind::Star => Some(BinaryOp::Mul),
            TokenKind::Slash => Some(BinaryOp::Div),
            TokenKind::Percent => Some(BinaryOp::Mod),
            
            TokenKind::Eq => Some(BinaryOp::Eq),
            TokenKind::Ne => Some(BinaryOp::Ne),
            TokenKind::Lt => Some(BinaryOp::Lt),
            TokenKind::Le => Some(BinaryOp::Le),
            TokenKind::Gt => Some(BinaryOp::Gt),
            TokenKind::Ge => Some(BinaryOp::Ge),
            TokenKind::Assign => Some(BinaryOp::Assign),

            TokenKind::And => Some(BinaryOp::And),
            TokenKind::Or => Some(BinaryOp::Or),
            _ => None,
        }
    }

    fn parse_match_expression(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 5);
        self.expect(TokenKind::Match)?;
        
        let scrutinee = Box::new(self.parse_expression()?);
        
        self.expect(TokenKind::LBrace)?;
        
        let mut arms = Vec::new();
        
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let pattern = self.parse_pattern()?;
            
            let guard = if self.match_any(&[TokenKind::If]) {
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            self.expect(TokenKind::FatArrow)?;
            
            let body = self.parse_expression()?;
            
            arms.push(crate::ast::node::MatchArm {
                pattern,
                guard,
                body,
                span,
            });
            
            if !self.check(TokenKind::RBrace) {
                self.match_any(&[TokenKind::Comma]);
            }
        }
        
        self.expect(TokenKind::RBrace)?;
        
        Ok(Expression::Match {
            scrutinee,
            arms,
            span,
        })
    }

    fn parse_block_expression(&mut self) -> ParseResult<Expression> {
        let block = self.parse_block()?;
        let span = block.span;
        Ok(Expression::Block { block, span })
    }

    fn parse_unary_expression(&mut self) -> ParseResult<Expression> {
        let span = Span::new(self.current.line, self.current.column, 1);
        let op = match self.current.kind {
            TokenKind::Minus => UnaryOp::Neg,
            TokenKind::Not => UnaryOp::Not,
            TokenKind::BitNot => UnaryOp::BitNot,
            _ => return Err(ParseError::UnexpectedToken { expected: TokenKind::Minus, found: self.current.kind.clone(), line: self.current.line, column: self.current.column }),
        };
        
        self.advance()?;
        let operand = self.parse_expression_with_precedence(Precedence::Prefix)?;
        
        Ok(Expression::Unary {
            op,
            operand: Box::new(operand),
            span,
        })
    }

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        let span = Span::new(self.current.line, self.current.column, 1);
        
        match &self.current.kind {
            TokenKind::Identifier => {
                let span = Span::new(self.current.line, self.current.column, 0);
                let name = self.current.text().unwrap_or("").to_string();
                
                if self.peek_token().kind == TokenKind::ColonColon {
                    let enum_ident = Identifier::new(name.clone(), span);
                    self.advance()?;
                    self.expect(TokenKind::ColonColon)?;
                    let vtok = self.expect(TokenKind::Identifier)?;
                    let vspan = Span::new(vtok.line, vtok.column, 0);
                    let variant_ident = Identifier::new(vtok.text.unwrap_or_default(), vspan);

                    let mut args: Vec<Pattern> = Vec::new();

                    if self.check(TokenKind::LParen) {
                        self.advance()?;
                        if !self.check(TokenKind::RParen) {
                            loop {
                                args.push(self.parse_pattern()?);
                                if !self.match_any(&[TokenKind::Comma]) { break; }
                            }
                        }
                        self.expect(TokenKind::RParen)?;

                        return Ok(Pattern::Enum { name: enum_ident, variant: variant_ident, args, named_fields: None, span });
                    }

                    else if self.check(TokenKind::LBrace) {
                        self.advance()?; // '{'

                        let mut fields = Vec::new();

                        if !self.check(TokenKind::RBrace) {
                            loop {
                                let ftok = self.expect(TokenKind::Identifier)?;
                                let fspan = Span::new(ftok.line, ftok.column, 0);
                                let fname = Identifier::new(ftok.text.unwrap_or_default(), fspan);

                                let pat = if self.match_any(&[TokenKind::Colon]) {
                                    Some(self.parse_pattern()?)
                                } else {
                                    None
                                };

                                fields.push(crate::ast::node::FieldPattern {
                                    name: fname,
                                    pattern: pat,
                                    span: fspan,
                                });

                                if !self.match_any(&[TokenKind::Comma]) { break; }
                            }
                        }

                        self.expect(TokenKind::RBrace)?;

                        return Ok(Pattern::Enum {
                            name: enum_ident,
                            variant: variant_ident,
                            args: Vec::new(),
                            named_fields: Some(fields),
                            span,
                        });
                    }

                    return Ok(Pattern::Enum { name: enum_ident, variant: variant_ident, args, named_fields: None, span });
                }
                
                if self.peek_token().kind == TokenKind::LBrace {
                     let struct_ident = Identifier::new(name.clone(), span);
                     self.advance()?;
                     self.expect(TokenKind::LBrace)?;
                     
                     let mut fields = Vec::new();
                     
                     if !self.check(TokenKind::RBrace) {
                         loop {
                             let ftok = self.expect(TokenKind::Identifier)?;
                             let fspan = Span::new(ftok.line, ftok.column, 0);
                             let fname = Identifier::new(ftok.text.unwrap_or_default(), fspan);
                             
                             let pat = if self.match_any(&[TokenKind::Colon]) {
                                 Some(self.parse_pattern()?) 
                             } else {
                                 None
                             };
                             
                             fields.push(crate::ast::node::FieldPattern {
                                 name: fname,
                                 pattern: pat,
                                 span: fspan,
                             });
                             
                             if !self.match_any(&[TokenKind::Comma]) { break; }
                         }
                     }
                     self.expect(TokenKind::RBrace)?;
                     
                     return Ok(Pattern::Struct {
                         name: struct_ident,
                         fields,
                         span,
                     });
                }

                if name == "Some" {
                    self.advance()?;
                    self.expect(TokenKind::LParen)?;
                    let inner_pat = self.parse_pattern()?;
                    self.expect(TokenKind::RParen)?;
                    return Ok(Pattern::Some {
                        pattern: Box::new(inner_pat),
                        span,
                    });
                }
                
                if name == "None" {
                    self.advance()?;
                    return Ok(Pattern::None { span });
                }

                if name == "Ok" {
                    self.advance()?;
                    self.expect(TokenKind::LParen)?;
                    let inner_pat = self.parse_pattern()?;
                    self.expect(TokenKind::RParen)?;
                    return Ok(Pattern::Ok {
                        pattern: Box::new(inner_pat),
                        span,
                    });
                }
                
                if name == "Err" {
                    self.advance()?;
                    self.expect(TokenKind::LParen)?;
                    let inner_pat = self.parse_pattern()?;
                    self.expect(TokenKind::RParen)?;
                    return Ok(Pattern::Err {
                        pattern: Box::new(inner_pat),
                        span,
                    });
                }
                
                let identifier = Identifier::new(name, span);
                self.advance()?;
                Ok(Pattern::Identifier { name: identifier, span })
            }
            
            TokenKind::IntLiteral => {
                let text = self.current.text().unwrap_or("0");
                let value = text.parse().unwrap_or(0);
                self.advance()?;
                Ok(Pattern::Literal {
                    value: Literal::Int(value),
                    span,
                })
            }
            
            TokenKind::True | TokenKind::False => {
                let value = self.current.kind == TokenKind::True;
                self.advance()?;
                Ok(Pattern::Literal {
                    value: Literal::Bool(value),
                    span,
                })
            }
            
            TokenKind::StringLiteral => {
                let text = self.current.text().unwrap_or("").to_string();
                self.advance()?;
                Ok(Pattern::Literal {
                    value: Literal::String(text),
                    span,
                })
            }
            
            TokenKind::LParen => {
                self.advance()?;
                let mut patterns = Vec::new();
                
                while !self.check(TokenKind::RParen) && !self.is_at_end() {
                    patterns.push(self.parse_pattern()?);
                    
                    if !self.match_any(&[TokenKind::Comma]) {
                        break;
                    }
                }
                
                self.expect(TokenKind::RParen)?;
                
                Ok(Pattern::Tuple { patterns, span })
            }
            
            _ => Err(ParseError::ExpectedPattern {
                found: self.current.kind,
                line: self.current.line,
                column: self.current.column,
            }),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> ParseResult<Program> {
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer)?;
        parser.parse_program()
    }

    #[test]
    fn test_parse_empty_function() {
        let input = "fn main() {}";
        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_function_with_params() {
        let input = "fn add(x: Int, y: Int) -> Int { return x; }";
        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_struct() {
        let input = "struct Point { x: Float, y: Float }";
        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_let_statement() {
        let input = "fn main() { let x: Int = 42; }";
        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_expression() {
        let input = "fn main() { 1 + 2 * 3; }";
        let result = parse(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_extern() {
        let input = r#"
            extern "C" {
                fn add(a: Int, b: Int) -> Int;
                async fn fetch(url: String) -> String;
            }
        "#;
        let result = parse(input);
        assert!(result.is_ok());
        let prog = result.unwrap();
        assert_eq!(prog.items.len(), 1);
        if let Item::Extern(ext) = &prog.items[0] {
            assert_eq!(ext.abi, "C");
            assert_eq!(ext.functions.len(), 2);
            assert_eq!(ext.functions[0].name.name, "add");
            assert_eq!(ext.functions[1].name.name, "fetch");
            assert!(ext.functions[1].is_async);
        } else {
            panic!("Expected Extern item");
        }
    }

    #[test]
    fn test_parse_attributes() {
        let input = r#"
            #[derive(Clone, Debug)]
            #[link(name = "m")]
            #[differentiable]
            fn test() {}
        "#;
        let result = parse(input);
        assert!(result.is_ok());
        let prog = result.unwrap();
        if let Item::Function(f) = &prog.items[0] {
            assert_eq!(f.attributes.len(), 3);
            match &f.attributes[0] {
                Attribute::Derive(traits) => assert_eq!(traits.len(), 2),
                _ => panic!("Expected Derive attribute"),
            }
            match &f.attributes[1] {
                Attribute::Nested(id, inner) => {
                    assert_eq!(id.name, "link");
                    assert_eq!(inner.len(), 1);
                    match &inner[0] {
                        Attribute::Value(inner_id, Literal::String(val)) => {
                            assert_eq!(inner_id.name, "name");
                            assert_eq!(val, "m");
                        }
                        _ => panic!("Expected Value attribute inside link"),
                    }
                }
                _ => panic!("Expected Nested attribute for link(...)"),
            }
            match &f.attributes[2] {
                Attribute::Simple(id) => assert_eq!(id.name, "differentiable"),
                _ => panic!("Expected Simple attribute"),
            }
        }
    }
}
