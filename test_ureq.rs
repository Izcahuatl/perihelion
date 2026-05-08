fn main() { let r = ureq::get("http://example.com").call().unwrap(); let _ = r.into_body().into_reader(); }
