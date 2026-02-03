use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Opcode {
    Const,
    Pop,
    Dup,
    
    GetLocal,
    SetLocal,
    GetGlobal,
    SetGlobal,
    DefineGlobal,
    
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    
    And,
    Or,
    Not,
    
    Jump,
    JumpIfFalse,
    JumpIfTrue,
    Loop,
    
    Call,
    Return,
    
    Closure,
    GetUpvalue,
    SetUpvalue,
    CloseUpvalue,
    
    GetField,
    SetField,
    MakeStruct,
    MakeArray,
    MakeEnum,
    Index,
    SetIndex,

    CloseScope,
    
    Sample,
    Observe,
    MakeGaussian,
    MakeUniform,
    MakeBernoulli,
    MakeBeta,
    
    MethodCall,
    DefineMethod,
    Print,
    Nil,
    True,
    False,

    IsVariant,
    UnpackEnum,
    IsType,
    
    Iterator,
    Next,
    MakeRange,
}

impl Opcode {
    pub fn operand_count(self) -> usize {
        match self {
            Opcode::Const | Opcode::GetLocal | Opcode::SetLocal |
            Opcode::GetGlobal | Opcode::SetGlobal | Opcode::DefineGlobal |
            Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue |
            Opcode::Loop | Opcode::Closure | Opcode::GetUpvalue |
            Opcode::SetUpvalue | Opcode::GetField | Opcode::SetField |
            Opcode::MakeArray | Opcode::CloseScope => 2,
            
            Opcode::Call | Opcode::MethodCall => 1,
            
            Opcode::MakeStruct | Opcode::MakeEnum => 3,
            
            Opcode::Next => 2,
            Opcode::MakeRange => 1,
            
            _ => 0,
        }
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub opcode: Opcode,
    pub operands: Vec<u8>,
    pub line: usize,
}

impl Instruction {
    pub fn simple(opcode: Opcode, line: usize) -> Self {
        Instruction { opcode, operands: vec![], line }
    }
    
    pub fn with_u8(opcode: Opcode, operand: u8, line: usize) -> Self {
        Instruction { opcode, operands: vec![operand], line }
    }
    
    pub fn with_u16(opcode: Opcode, operand: u16, line: usize) -> Self {
        let bytes = operand.to_le_bytes();
        Instruction { opcode, operands: vec![bytes[0], bytes[1]], line }
    }
    
    pub fn with_i16(opcode: Opcode, operand: i16, line: usize) -> Self {
        let bytes = operand.to_le_bytes();
        Instruction { opcode, operands: vec![bytes[0], bytes[1]], line }
    }
    
    pub fn read_u16(&self) -> u16 {
        u16::from_le_bytes([self.operands[0], self.operands[1]])
    }
    
    pub fn read_i16(&self) -> i16 {
        i16::from_le_bytes([self.operands[0], self.operands[1]])
    }
    
    pub fn read_u8(&self) -> u8 {
        self.operands[0]
    }
}
