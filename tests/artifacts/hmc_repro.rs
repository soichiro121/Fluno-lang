
use fluno::vm::{Interpreter, Value};
use fluno::vm::prob_runtime::HMC;
use fluno::parser::Parser;
use fluno::lexer::Lexer;
use fluno::ast::node::{Statement, Item};
use std::rc::Rc;

#[test]
fn test_hmc_reproduction() {
    // 1. Setup Interpreter
    let mut interpreter = Interpreter::new();
    // interpreter.init_builtins(); // Already called in new()

    // 2. Define a simple model: Prior mu ~ N(0, 10), Observation 5.0 ~ N(mu, 1)
    // We'll parse this from strict generic Fluno syntax
    let source = r#"
        let mu = sample(Gaussian(0.0, 10.0));
        observe(Gaussian(mu, 1.0), 5.0);
        mu
    "#;

    let mut parser = Parser::new(Lexer::new(source));
    // Usually parser returns a Program (list of items).
    // But here we just want statements. Fluno parser might expect top-level items.
    // Let's wrap it in a function if needed, or use a specific parsing method.
    // Assuming we can parse statements:
    // Actually, let's just make a Program with a function, and extract the body.
    
    let source_wrapper = r#"
        fn model() {
            let mu = sample(Gaussian(0.0, 10.0));
            observe(Gaussian(mu, 1.0), 5.0);
            mu
        }
    "#;
    let mut parser = Parser::new(Lexer::new(source_wrapper)).expect("Failed to create parser");
    let program = parser.parse_program().expect("Failed to parse");

    // Extract body of 'model'
    let model_block = match &program.items[0] {
        Item::Function(func) => func.body.clone(),
        _ => panic!("Expected function"),
    };

    // 3. Setup HMC
    // HMC::new(num_samples, burn_in, epsilon, l_steps)
    let hmc = HMC::new(10, 0, 0.1, 5);

    // 4. Run Inference
    use std::collections::HashMap;
    let result = hmc.infer(&model_block, &mut interpreter, HashMap::new());

    // 5. Check result
    // Current BROKEN behavior: returns Vec<Value::Float> (single values) but arbitrary first param?
    // And uses incorrect momentum.
    
    match result {
        Ok(samples) => {
            println!("Got {} samples", samples.len());
            for s in samples {
                println!("Sample: {:?}", s);
            }
        },
        Err(e) => panic!("HMC failed: {:?}", e),
    }
}
