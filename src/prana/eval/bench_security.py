"""Security benchmark (design doc FR-07: "flag common vulnerabilities: SQL
injection, hardcoded secrets, insecure deserialization, missing input
validation, path traversal"): 33 vulnerable/fixed pairs across 5 languages.

The 13 security cases of `bench_cases` plus 20 more below, each tagged with
its vulnerability class. Same shape as `bench_cases.CASES`, so the same runner
and judge measure detection (on `bad`) and false positives (on `good`).
"""

from __future__ import annotations

from prana.eval.bench_cases import CASES as _BASE

KIND_OF = {  # vulnerability class of the 13 shared cases
    "py-sqli": "injection", "py-shell": "injection", "py-pickle": "deserialization",
    "py-traversal": "path-traversal", "ts-xss": "xss", "ts-secret": "secrets",
    "js-eval": "injection", "js-sqli": "injection", "js-open-redirect": "redirect",
    "go-sqli": "injection", "go-cmd-injection": "injection", "rs-sqli": "injection",
    "rs-secret": "secrets",
}

EXTRA: list[dict] = [
    # ------------------------------------------------------------------ python
    dict(id="py-yaml-load", language="python", path="app/importer.py", category="SECURITY",
         kind="deserialization",
         issue="yaml.load with the full Loader on uploaded content can construct arbitrary Python objects; use yaml.safe_load.",
         bad='''def import_settings(uploaded: bytes) -> dict:
    return yaml.load(uploaded, Loader=yaml.Loader)''',
         good='''def import_settings(uploaded: bytes) -> dict:
    return yaml.safe_load(uploaded)'''),
    dict(id="py-md5-password", language="python", path="app/auth.py", category="SECURITY",
         kind="crypto",
         issue="Passwords hashed with unsalted MD5 are trivially cracked; use a slow salted hash such as bcrypt, scrypt or argon2.",
         bad='''def hash_password(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()''',
         good='''def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()'''),
    dict(id="py-verify-false", language="python", path="app/billing.py", category="SECURITY",
         kind="tls",
         issue="verify=False disables TLS certificate checks, so the payment call can be intercepted; keep verification on.",
         bad='''def charge(card_token: str, amount: int) -> dict:
    r = requests.post(PAY_URL, json={"token": card_token, "amount": amount}, verify=False, timeout=10)
    return r.json()''',
         good='''def charge(card_token: str, amount: int) -> dict:
    r = requests.post(PAY_URL, json={"token": card_token, "amount": amount}, timeout=10)
    return r.json()'''),
    dict(id="py-ssrf", language="python", path="app/preview.py", category="SECURITY",
         kind="ssrf",
         issue="SSRF: the server fetches any URL the user supplies, including internal addresses; restrict scheme and host to an allow-list.",
         bad='''def link_preview(request):
    url = request.args["url"]
    return requests.get(url, timeout=5).text[:500]''',
         good='''def link_preview(request):
    url = request.args["url"]
    host = urllib.parse.urlparse(url).hostname
    if not url.startswith("https://") or host not in ALLOWED_PREVIEW_HOSTS:
        raise ValueError("url not allowed")
    return requests.get(url, timeout=5).text[:500]'''),
    dict(id="py-unbounded-limit", language="python", path="app/api.py", category="SECURITY",
         kind="validation",
         issue="Missing input validation: the client controls 'limit' with no upper bound, so one request can load the whole table; clamp it.",
         bad='''def list_orders(request, db):
    limit = int(request.args.get("limit", 50))
    return db.query(Order).limit(limit).all()''',
         good='''def list_orders(request, db):
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    return db.query(Order).limit(limit).all()'''),

    # -------------------------------------------------------------- typescript
    dict(id="ts-jwt-decode", language="typescript", path="src/auth.ts", category="SECURITY",
         kind="auth",
         issue="jwt.decode does not verify the signature, so anyone can forge a token; use jwt.verify with the secret.",
         bad='''export function currentUser(token: string): Claims {
  return jwt.decode(token) as Claims;
}''',
         good='''export function currentUser(token: string): Claims {
  return jwt.verify(token, process.env.JWT_SECRET as string, { algorithms: ["HS256"] }) as Claims;
}'''),
    dict(id="ts-sendfile-traversal", language="typescript", path="src/download.ts", category="SECURITY",
         kind="path-traversal",
         issue="Path traversal: the file name from the URL is joined to the upload directory unchecked, so ../ reaches any file.",
         bad='''export function download(req: Request, res: Response): void {
  res.sendFile(path.join(UPLOAD_DIR, req.params.name));
}''',
         good='''export function download(req: Request, res: Response): void {
  const file = path.resolve(UPLOAD_DIR, req.params.name);
  if (!file.startsWith(path.resolve(UPLOAD_DIR) + path.sep)) {
    res.status(400).end();
    return;
  }
  res.sendFile(file);
}'''),
    dict(id="ts-cookie-flags", language="typescript", path="src/session.ts", category="SECURITY",
         kind="auth",
         issue="The session cookie is readable by JavaScript and sent over plain HTTP; set httpOnly, secure and sameSite.",
         bad='''export function setSession(res: Response, sid: string): void {
  res.cookie("sid", sid);
}''',
         good='''export function setSession(res: Response, sid: string): void {
  res.cookie("sid", sid, { httpOnly: true, secure: true, sameSite: "lax" });
}'''),
    dict(id="ts-ssrf", language="typescript", path="src/webhook.ts", category="SECURITY",
         kind="ssrf",
         issue="SSRF: the server calls whatever callback URL the client registers, including internal services; validate against an allow-list.",
         bad='''export async function ping(callbackUrl: string): Promise<number> {
  const r = await fetch(callbackUrl, { method: "POST" });
  return r.status;
}''',
         good='''export async function ping(callbackUrl: string): Promise<number> {
  const u = new URL(callbackUrl);
  if (u.protocol !== "https:" || !ALLOWED_HOSTS.has(u.hostname)) {
    throw new Error("callback host not allowed");
  }
  const r = await fetch(u, { method: "POST" });
  return r.status;
}'''),

    # -------------------------------------------------------------- javascript
    dict(id="js-exec", language="javascript", path="lib/thumbs.js", category="SECURITY",
         kind="injection",
         issue="Command injection: the uploaded file name is interpolated into a shell command run by exec; use execFile with an argument array.",
         bad='''export function makeThumbnail(fileName, cb) {
  exec(`convert uploads/${fileName} -resize 128x128 thumbs/${fileName}`, cb);
}''',
         good='''export function makeThumbnail(fileName, cb) {
  execFile("convert", [`uploads/${fileName}`, "-resize", "128x128", `thumbs/${fileName}`], cb);
}'''),
    dict(id="js-random-token", language="javascript", path="lib/reset.js", category="SECURITY",
         kind="randomness",
         issue="Math.random is not cryptographically secure, so password-reset tokens can be predicted; use crypto.randomBytes.",
         bad='''export function resetToken() {
  return Math.random().toString(36).slice(2);
}''',
         good='''export function resetToken() {
  return crypto.randomBytes(32).toString("hex");
}'''),
    dict(id="js-hardcoded-password", language="javascript", path="lib/db.js", category="SECURITY",
         kind="secrets",
         issue="The database password is hard-coded in source; read it from the environment.",
         bad='''export const pool = new Pool({
  host: "db.internal",
  user: "shop",
  password: "Sh0p-Prod-2026!",
});''',
         good='''export const pool = new Pool({
  host: "db.internal",
  user: "shop",
  password: process.env.DB_PASSWORD,
});'''),
    dict(id="js-innerhtml", language="javascript", path="lib/search.js", category="SECURITY",
         kind="xss",
         issue="XSS: the search term from the URL is written into the page with innerHTML; use textContent.",
         bad='''export function showQuery(el) {
  const q = new URLSearchParams(location.search).get("q");
  el.innerHTML = "Results for " + q;
}''',
         good='''export function showQuery(el) {
  const q = new URLSearchParams(location.search).get("q");
  el.textContent = "Results for " + q;
}'''),

    # ---------------------------------------------------------------------- go
    dict(id="go-tls-skip", language="go", path="client/tls.go", category="SECURITY",
         kind="tls",
         issue="InsecureSkipVerify disables certificate verification, allowing man-in-the-middle attacks.",
         bad='''func NewClient() *http.Client {
	tr := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
	return &http.Client{Transport: tr, Timeout: 10 * time.Second}
}''',
         good='''func NewClient() *http.Client {
	return &http.Client{Timeout: 10 * time.Second}
}'''),
    dict(id="go-path-traversal", language="go", path="web/files.go", category="SECURITY",
         kind="path-traversal",
         issue="Path traversal: the name from the query string is joined to the base directory unchecked; reject names that escape it.",
         bad='''func ServeReport(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	http.ServeFile(w, r, filepath.Join(reportDir, name))
}''',
         good='''func ServeReport(w http.ResponseWriter, r *http.Request) {
	name := filepath.Base(r.URL.Query().Get("name"))
	if name == "." || name == "/" {
		http.Error(w, "bad name", http.StatusBadRequest)
		return
	}
	http.ServeFile(w, r, filepath.Join(reportDir, name))
}'''),
    dict(id="go-math-rand-token", language="go", path="auth/token.go", category="SECURITY",
         kind="randomness",
         issue="math/rand is predictable; session tokens must come from crypto/rand.",
         bad='''func NewSessionToken() string {
	b := make([]byte, 32)
	for i := range b {
		b[i] = byte(rand.Intn(256))
	}
	return hex.EncodeToString(b)
}''',
         good='''func NewSessionToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}'''),
    dict(id="go-secret", language="go", path="config/keys.go", category="SECURITY",
         kind="secrets",
         issue="The signing key is hard-coded in source; load it from the environment or a secret store.",
         bad='''var signingKey = []byte("prod-signing-key-7f3a9c2b1e4d")

func Sign(msg []byte) []byte {
	m := hmac.New(sha256.New, signingKey)
	m.Write(msg)
	return m.Sum(nil)
}''',
         good='''func Sign(msg []byte) ([]byte, error) {
	key := os.Getenv("SIGNING_KEY")
	if key == "" {
		return nil, errors.New("SIGNING_KEY not set")
	}
	m := hmac.New(sha256.New, []byte(key))
	m.Write(msg)
	return m.Sum(nil), nil
}'''),

    # -------------------------------------------------------------------- rust
    dict(id="rs-path-traversal", language="rust", path="src/static_files.rs", category="SECURITY",
         kind="path-traversal",
         issue="Path traversal: the requested name is joined to the asset directory without rejecting '..' components.",
         bad='''pub fn asset_path(base: &Path, name: &str) -> PathBuf {
    base.join(name)
}''',
         good='''pub fn asset_path(base: &Path, name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    if p.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
        return None;
    }
    Some(base.join(p))
}'''),
    dict(id="rs-shell", language="rust", path="src/backup.rs", category="SECURITY",
         kind="injection",
         issue="Command injection: the archive name is formatted into a shell command; call tar directly with arguments.",
         bad='''pub fn backup(name: &str) -> std::io::Result<std::process::ExitStatus> {
    Command::new("sh").arg("-c").arg(format!("tar czf /backups/{}.tgz /data", name)).status()
}''',
         good='''pub fn backup(name: &str) -> std::io::Result<std::process::ExitStatus> {
    Command::new("tar").arg("czf").arg(format!("/backups/{}.tgz", name)).arg("/data").status()
}'''),
    dict(id="rs-accept-invalid-certs", language="rust", path="src/http.rs", category="SECURITY",
         kind="tls",
         issue="danger_accept_invalid_certs(true) turns off TLS certificate validation; remove it.",
         bad='''pub fn client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder().danger_accept_invalid_certs(true).build()
}''',
         good='''pub fn client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder().build()
}'''),
]

CASES: list[dict] = [{**c, "kind": KIND_OF[c["id"]]} for c in _BASE if c["category"] == "SECURITY"] \
    + EXTRA
