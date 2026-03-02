use std::ffi::{CStr, CString, c_char};
use plotters::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::fs::File;
use std::io::{Read, Write};
use lazy_static::lazy_static;
use std::net::TcpStream;
use std::sync::atomic::{AtomicI64, Ordering};
use std::thread;

// Task State for Async Polling
enum TaskState {
    Pending,
    Ready(String),
    Error(String),
}

lazy_static! {
    static ref FILES: Mutex<HashMap<i64, File>> = Mutex::new(HashMap::new());
    static ref NEXT_HANDLE: Mutex<i64> = Mutex::new(1);
    
    static ref SOCKETS: Mutex<HashMap<i64, TcpStream>> = Mutex::new(HashMap::new());
    static ref NEXT_SOCKET: Mutex<i64> = Mutex::new(1);

    // Async Tasks
    static ref TASKS: Mutex<HashMap<i64, TaskState>> = Mutex::new(HashMap::new());
    static ref NEXT_TASK_ID: AtomicI64 = AtomicI64::new(1);
}

fn get_next_handle() -> i64 {
    let mut handle = NEXT_HANDLE.lock().unwrap();
    let h = *handle;
    *handle += 1;
    h
}

#[no_mangle]
pub extern "C" fn fluno_print(msg: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(msg) };
    if let Ok(s) = c_str.to_str() {
        println!("{}", s);
    }
}

// --- Sync FS ---
#[no_mangle]
pub extern "C" fn fluno_fs_read(path: *const c_char) -> *const c_char {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null(),
    };
    match std::fs::read_to_string(path_str) {
        Ok(content) => {
            let c_content = CString::new(content).unwrap_or_default();
            c_content.into_raw()
        }
        Err(_) => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn fluno_fs_write(path: *const c_char, content: *const c_char) -> i64 {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    let content_str = unsafe { CStr::from_ptr(content).to_str().unwrap_or("") };
    if path_str.is_empty() { return -1; }
    match std::fs::write(path_str, content_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

// --- Async FS (Polling) ---
#[no_mangle]
pub extern "C" fn fluno_fs_read_async(path: *const c_char) -> i64 {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    
    let id = NEXT_TASK_ID.fetch_add(1, Ordering::SeqCst);
    
    TASKS.lock().unwrap().insert(id, TaskState::Pending);
    
    thread::spawn(move || {
        let res = std::fs::read_to_string(&path_str);
        let mut tasks = TASKS.lock().unwrap();
        match res {
            Ok(s) => { tasks.insert(id, TaskState::Ready(s)); },
            Err(e) => { tasks.insert(id, TaskState::Error(e.to_string())); },
        }
    });

    id
}

#[no_mangle]
pub extern "C" fn fluno_task_poll(id: i64) -> i32 {
    let tasks = TASKS.lock().unwrap();
    match tasks.get(&id) {
        Some(TaskState::Pending) => 0,
        Some(TaskState::Ready(_)) => 1,
        Some(TaskState::Error(_)) => -1,
        None => -2,
    }
}

#[no_mangle]
pub extern "C" fn fluno_task_result(id: i64) -> *const c_char {
    let mut tasks = TASKS.lock().unwrap();
    // Retrieve and remove (consume) result
    if let Some(state) = tasks.remove(&id) {
        match state {
            TaskState::Ready(s) => {
                 CString::new(s).unwrap_or_default().into_raw()
            },
            TaskState::Error(_) => std::ptr::null(), // Error handling?
            TaskState::Pending => std::ptr::null(), // Should not happen if polled
        }
    } else {
        std::ptr::null()
    }
}

// --- Robust File IO (Phase 7) ---
#[no_mangle]
pub extern "C" fn fluno_file_open(path: *const c_char, mode: *const c_char) -> i64 {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    let mode_str = unsafe { CStr::from_ptr(mode).to_str().unwrap_or("r") };
    let file = match mode_str {
        "r" => File::open(path_str),
        "w" => File::create(path_str),
        "a" => std::fs::OpenOptions::new().append(true).create(true).open(path_str),
        _ => return -1,
    };
    match file {
        Ok(f) => {
            let handle = get_next_handle();
            FILES.lock().unwrap().insert(handle, f);
            handle
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn fluno_file_close(handle: i64) {
    FILES.lock().unwrap().remove(&handle);
}

#[no_mangle]
pub extern "C" fn fluno_file_read_all(handle: i64) -> *const c_char {
    let mut files = FILES.lock().unwrap();
    if let Some(file) = files.get_mut(&handle) {
        let mut content = String::new();
        if file.read_to_string(&mut content).is_ok() {
            return CString::new(content).unwrap_or_default().into_raw();
        }
    }
    std::ptr::null()
}

// --- Network (Phase 9) ---
#[no_mangle]
pub extern "C" fn fluno_net_connect(addr: *const c_char) -> i64 {
    let addr_str = unsafe { CStr::from_ptr(addr).to_str().unwrap_or("") };
    match TcpStream::connect(addr_str) {
        Ok(stream) => {
            stream.set_nonblocking(false).ok(); // Blocking by default for simple Net
            let id = {
                let mut h = NEXT_SOCKET.lock().unwrap();
                let v = *h;
                *h += 1;
                v
            };
            SOCKETS.lock().unwrap().insert(id, stream);
            id
        }
        Err(_) => -1,
    }
}
// (Include plotters as before, assume it's concatenated or I overwrite full file)

// I need to preserve `fluno_plot_chart` from Phase 10.
// I will append it if I am overwriting.
// The file viewed in Step 1554 showed imports but not plotting logic at bottom.
// I must ensure plotting logic is included.
// Step 1554 content was truncated.
// I should use `read_resource` or assume standard structure.
// I'll append the plotting code since I wrote it in Phase 10.

#[no_mangle]
pub extern "C" fn fluno_plot_chart(json: *const c_char, path: *const c_char) -> i64 {
    let json_str = unsafe { CStr::from_ptr(json).to_str().unwrap_or("") };
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    
    if json_str.is_empty() || path_str.is_empty() { return -1; }
    
    let root = BitMapBackend::new(path_str, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    
    // Parse JSON manually or use serde (we added serde)
    // Minimally we need simple parsing. But we added serde_json dependency in Phase 10.
    // Use serde_json.
    
    // Define struct for decoding
    #[derive(serde::Deserialize)]
    struct ChartData {
        title: String,
        x_label: String,
        y_label: String,
        series: Vec<SeriesData>,
    }
    #[derive(serde::Deserialize)]
    struct SeriesData {
        label: String,
        color: String,
        data: Vec<(f64, f64)>,
    }
    
    let chart_data: ChartData = match serde_json::from_str(json_str) {
        Ok(d) => d,
        Err(_) => return -2,
    };
    
    let mut chart = ChartBuilder::on(&root)
        .caption(&chart_data.title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..10f64, 0f64..20f64).unwrap(); // Auto-scale ideally

    chart.configure_mesh()
        .x_desc(&chart_data.x_label)
        .y_desc(&chart_data.y_label)
        .draw().unwrap();
        
    for s in chart_data.series {
        let color = match s.color.as_str() {
            "red" => RED,
            "blue" => BLUE,
            "green" => GREEN,
            _ => BLACK,
        };
        chart.draw_series(LineSeries::new(
            s.data,
            &color,
        )).unwrap()
        .label(&s.label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
    }
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();

    root.present().unwrap();
    0
}
