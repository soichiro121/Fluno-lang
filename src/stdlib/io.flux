
#[link(name = "c:/Users/froms/Desktop/Flux-main/fluno-rt/target/release/fluno_rt.dll")]
extern "C" {
     fn fluno_fs_read(path: String) -> String;
     fn fluno_fs_write(path: String, content: String) -> Int;
     fn fluno_time_now() -> Int;
     fn fluno_time_sleep(ms: Int) -> Int;
}

// User-friendly Wrappers

fn read_file(path: String) -> String {
    // In a real stdlib, we would return Result<String, Error>
    fluno_fs_read(path)
}

fn write_file(path: String, content: String) -> Bool {
    let result = fluno_fs_write(path, content);
    if result == 0 {
        true
    } else {
        false
    }
}

fn now() -> Int {
    fluno_time_now()
}

fn sleep(ms: Int) {
    let _ = fluno_time_sleep(ms);
}
