use std::error::Error;

fn main() -> Result<(), impl Error> {
    println!("Hello, world!");

    std::io::Result::Ok(())
}
