"""Planted-defect benchmark v1: 40 pairs, 8 per language.

Each case is the same small change written twice: `bad` contains one known
defect, `good` is the corrected version with nothing else changed that a
careful reviewer should object to. Reviewing `bad` measures issue detection;
reviewing `good` measures false positives (design doc §11).

Hand-written; every `good` was checked to be correct and idiomatic. Fields:
id, language, path, category (the model's issue types), issue (what a correct
review of `bad` must say), bad, good.
"""

from __future__ import annotations

CASES: list[dict] = [
    # ------------------------------------------------------------------ python
    dict(id="py-sqli", language="python", path="app/users.py", category="SECURITY",
         issue="SQL injection: the query is built with an f-string from user input; use a parameterised query.",
         bad='''def find_user(conn, name):
    sql = f"SELECT id, email FROM users WHERE name = '{name}'"
    return conn.execute(sql).fetchone()''',
         good='''def find_user(conn, name):
    sql = "SELECT id, email FROM users WHERE name = ?"
    return conn.execute(sql, (name,)).fetchone()'''),
    dict(id="py-shell", language="python", path="app/tools.py", category="SECURITY",
         issue="Command injection: shell=True with a user-supplied file name; pass an argument list without a shell.",
         bad='''def word_count(filename):
    out = subprocess.run(f"wc -w {filename}", shell=True, capture_output=True, text=True)
    return int(out.stdout.split()[0])''',
         good='''def word_count(filename):
    out = subprocess.run(["wc", "-w", filename], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])'''),
    dict(id="py-pickle", language="python", path="app/session.py", category="SECURITY",
         issue="Insecure deserialization: pickle.loads on data from the client allows remote code execution; use JSON (and sign it).",
         bad='''def read_session(request):
    raw = base64.b64decode(request.cookies["session"])
    return pickle.loads(raw)''',
         good='''def read_session(request):
    raw = base64.b64decode(request.cookies["session"])
    return json.loads(raw)'''),
    dict(id="py-traversal", language="python", path="app/files.py", category="SECURITY",
         issue="Path traversal: a user-supplied name like ../../etc/passwd escapes the base directory; resolve the path and check it stays inside.",
         bad='''def read_report(name):
    path = os.path.join(REPORT_DIR, name)
    with open(path, "rb") as f:
        return f.read()''',
         good='''def read_report(name):
    base = Path(REPORT_DIR).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError("invalid report name")
    return path.read_bytes()'''),
    dict(id="py-mutable-default", language="python", path="app/cart.py", category="BUG",
         issue="Mutable default argument: the same list is shared across calls, so items leak between carts; default to None.",
         bad='''def add_item(item, cart=[]):
    cart.append(item)
    return cart''',
         good='''def add_item(item, cart=None):
    cart = [] if cart is None else cart
    cart.append(item)
    return cart'''),
    dict(id="py-off-by-one", language="python", path="app/stats.py", category="BUG",
         issue="Off-by-one: range(len(values) - 1) skips the last value, so the total is wrong.",
         bad='''def total(values):
    s = 0
    for i in range(len(values) - 1):
        s += values[i]
    return s''',
         good='''def total(values):
    s = 0
    for v in values:
        s += v
    return s'''),
    dict(id="py-leak", language="python", path="app/config.py", category="BUG",
         issue="Resource leak: the file is opened and never closed; use a with block.",
         bad='''def load_config(path):
    f = open(path, encoding="utf-8")
    return json.load(f)''',
         good='''def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)'''),
    dict(id="py-quadratic", language="python", path="app/report.py", category="PERF",
         issue="Quadratic time: membership tests on a list inside the loop; use a set.",
         bad='''def unique_emails(users):
    seen = []
    for u in users:
        if u.email not in seen:
            seen.append(u.email)
    return seen''',
         good='''def unique_emails(users):
    seen = set()
    out = []
    for u in users:
        if u.email not in seen:
            seen.add(u.email)
            out.append(u.email)
    return out'''),

    # -------------------------------------------------------------- typescript
    dict(id="ts-missing-await", language="typescript", path="src/orders.ts", category="BUG",
         issue="Missing await: the insert promise is not awaited, so errors are lost and the caller is told it succeeded.",
         bad='''export async function saveOrder(db: Db, order: Order): Promise<string> {
  db.orders.insert(order);
  return "saved";
}''',
         good='''export async function saveOrder(db: Db, order: Order): Promise<string> {
  await db.orders.insert(order);
  return "saved";
}'''),
    dict(id="ts-optional", language="typescript", path="src/profile.ts", category="BUG",
         issue="profile is optional, so user.profile.name throws for users without a profile; use optional chaining with a fallback.",
         bad='''export function displayName(user: { name: string; profile?: { nick?: string } }): string {
  return user.profile.nick || user.name;
}''',
         good='''export function displayName(user: { name: string; profile?: { nick?: string } }): string {
  return user.profile?.nick || user.name;
}'''),
    dict(id="ts-xss", language="typescript", path="src/comments.ts", category="SECURITY",
         issue="XSS: user-written comment text is assigned to innerHTML; use textContent.",
         bad='''export function renderComment(el: HTMLElement, comment: { author: string; text: string }): void {
  el.innerHTML = `<b>${comment.author}</b>: ${comment.text}`;
}''',
         good='''export function renderComment(el: HTMLElement, comment: { author: string; text: string }): void {
  const b = document.createElement("b");
  b.textContent = comment.author;
  el.replaceChildren(b, document.createTextNode(`: ${comment.text}`));
}'''),
    dict(id="ts-nan", language="typescript", path="src/parse.ts", category="BUG",
         issue="Comparing to NaN with === is always false, so invalid input is never rejected; use Number.isNaN.",
         bad='''export function parseQuantity(raw: string): number {
  const n = Number(raw);
  if (n === NaN) {
    throw new Error("not a number");
  }
  return n;
}''',
         good='''export function parseQuantity(raw: string): number {
  const n = Number(raw);
  if (Number.isNaN(n)) {
    throw new Error("not a number");
  }
  return n;
}'''),
    dict(id="ts-sort-mutates", language="typescript", path="src/table.ts", category="BUG",
         issue="Array.prototype.sort sorts in place, so this mutates the caller's array (props); copy before sorting.",
         bad='''export function sortedByPrice(items: Item[]): Item[] {
  return items.sort((a, b) => a.price - b.price);
}''',
         good='''export function sortedByPrice(items: Item[]): Item[] {
  return [...items].sort((a, b) => a.price - b.price);
}'''),
    dict(id="ts-sequential", language="typescript", path="src/prices.ts", category="PERF",
         issue="Independent requests are awaited one after another inside the loop; run them concurrently with Promise.all.",
         bad='''export async function fetchPrices(ids: string[]): Promise<number[]> {
  const prices: number[] = [];
  for (const id of ids) {
    prices.push(await getPrice(id));
  }
  return prices;
}''',
         good='''export async function fetchPrices(ids: string[]): Promise<number[]> {
  return Promise.all(ids.map((id) => getPrice(id)));
}'''),
    dict(id="ts-secret", language="typescript", path="src/payments.ts", category="SECURITY",
         issue="A live payment API key is hard-coded in source; read it from the environment or a secret store.",
         bad='''const PAYMENT_KEY = "demo-live-payment-key-7f3a9c2b1e4d";

export function paymentHeaders(): Record<string, string> {
  return { Authorization: `Bearer ${PAYMENT_KEY}` };
}''',
         good='''export function paymentHeaders(): Record<string, string> {
  const key = process.env.PAYMENT_KEY;
  if (!key) throw new Error("PAYMENT_KEY is not set");
  return { Authorization: `Bearer ${key}` };
}'''),
    dict(id="ts-off-by-one", language="typescript", path="src/sum.ts", category="BUG",
         issue="Off-by-one: i <= values.length reads one past the end (undefined), producing NaN.",
         bad='''export function sum(values: number[]): number {
  let s = 0;
  for (let i = 0; i <= values.length; i++) {
    s += values[i];
  }
  return s;
}''',
         good='''export function sum(values: number[]): number {
  let s = 0;
  for (let i = 0; i < values.length; i++) {
    s += values[i];
  }
  return s;
}'''),

    # -------------------------------------------------------------- javascript
    dict(id="js-eval", language="javascript", path="lib/filters.js", category="SECURITY",
         issue="eval on a query-string value lets any visitor run code on the server; parse it with JSON.parse.",
         bad='''export function parseFilter(req) {
  return eval("(" + req.query.filter + ")");
}''',
         good='''export function parseFilter(req) {
  return JSON.parse(req.query.filter);
}'''),
    dict(id="js-sqli", language="javascript", path="lib/orders.js", category="SECURITY",
         issue="SQL injection: the order id from the URL is concatenated into the query; use a placeholder.",
         bad='''export async function getOrder(db, req) {
  const rows = await db.query("SELECT * FROM orders WHERE id = " + req.params.id);
  return rows[0];
}''',
         good='''export async function getOrder(db, req) {
  const rows = await db.query("SELECT * FROM orders WHERE id = $1", [req.params.id]);
  return rows[0];
}'''),
    dict(id="js-coercion", language="javascript", path="lib/stock.js", category="BUG",
         issue="Loose equality: an empty string == 0 is true, so a missing quantity is treated as out of stock; use === with a number.",
         bad='''export function isOutOfStock(qtyField) {
  return qtyField == 0;
}''',
         good='''export function isOutOfStock(qtyField) {
  return Number(qtyField) === 0 && qtyField !== "";
}'''),
    dict(id="js-month", language="javascript", path="lib/dates.js", category="BUG",
         issue="JavaScript months are zero-based: new Date(year, 12, 1) is January of the next year, not December.",
         bad='''export function decemberFirst(year) {
  return new Date(year, 12, 1);
}''',
         good='''export function decemberFirst(year) {
  return new Date(year, 11, 1);
}'''),
    dict(id="js-map-braces", language="javascript", path="lib/prices.js", category="BUG",
         issue="The arrow function has a block body without return, so map produces an array of undefined.",
         bad='''export function withTax(prices) {
  return prices.map((p) => { p * 1.18 });
}''',
         good='''export function withTax(prices) {
  return prices.map((p) => p * 1.18);
}'''),
    dict(id="js-foreach-async", language="javascript", path="lib/sync.js", category="BUG",
         issue="forEach does not wait for async callbacks, so 'done' is reported before the saves finish and errors are unhandled.",
         bad='''export async function saveAll(items, save) {
  items.forEach(async (item) => {
    await save(item);
  });
  return "done";
}''',
         good='''export async function saveAll(items, save) {
  for (const item of items) {
    await save(item);
  }
  return "done";
}'''),
    dict(id="js-open-redirect", language="javascript", path="lib/auth.js", category="SECURITY",
         issue="Open redirect: the post-login redirect target comes straight from the query string; allow only same-site relative paths.",
         bad='''export function afterLogin(req, res) {
  res.redirect(req.query.next || "/");
}''',
         good='''export function afterLogin(req, res) {
  const next = req.query.next;
  const safe = typeof next === "string" && next.startsWith("/") && !next.startsWith("//");
  res.redirect(safe ? next : "/");
}'''),
    dict(id="js-var-closure", language="javascript", path="lib/retry.js", category="BUG",
         issue="var is function-scoped, so every timeout logs the final value of i; use let.",
         bad='''export function scheduleRetries(n) {
  for (var i = 1; i <= n; i++) {
    setTimeout(() => console.log("retry", i), i * 1000);
  }
}''',
         good='''export function scheduleRetries(n) {
  for (let i = 1; i <= n; i++) {
    setTimeout(() => console.log("retry", i), i * 1000);
  }
}'''),

    # ---------------------------------------------------------------------- go
    dict(id="go-ignored-error", language="go", path="store/load.go", category="BUG",
         issue="The error from os.Open is discarded; on failure f is nil and the next call panics.",
         bad='''func LoadLines(path string) ([]string, error) {
	f, _ := os.Open(path)
	defer f.Close()
	return readLines(f)
}''',
         good='''func LoadLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", path, err)
	}
	defer f.Close()
	return readLines(f)
}'''),
    dict(id="go-defer-loop", language="go", path="store/merge.go", category="BUG",
         issue="defer inside the loop keeps every file open until the function returns; close each file in the iteration.",
         bad='''func CountAll(paths []string) (int, error) {
	total := 0
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return 0, err
		}
		defer f.Close()
		n, err := countLines(f)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}''',
         good='''func CountAll(paths []string) (int, error) {
	total := 0
	for _, p := range paths {
		n, err := countFile(p)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

func countFile(p string) (int, error) {
	f, err := os.Open(p)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return countLines(f)
}'''),
    dict(id="go-map-race", language="go", path="cache/hits.go", category="BUG",
         issue="Concurrent writes to a plain map from many goroutines are a data race (and can crash); guard it with a mutex.",
         bad='''func CountHits(urls []string) map[string]int {
	hits := map[string]int{}
	var wg sync.WaitGroup
	for _, u := range urls {
		wg.Add(1)
		go func(u string) {
			defer wg.Done()
			hits[u]++
		}(u)
	}
	wg.Wait()
	return hits
}''',
         good='''func CountHits(urls []string) map[string]int {
	hits := map[string]int{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, u := range urls {
		wg.Add(1)
		go func(u string) {
			defer wg.Done()
			mu.Lock()
			hits[u]++
			mu.Unlock()
		}(u)
	}
	wg.Wait()
	return hits
}'''),
    dict(id="go-sqli", language="go", path="store/users.go", category="SECURITY",
         issue="SQL injection: the email is formatted into the query with Sprintf; use a placeholder argument.",
         bad='''func FindUser(db *sql.DB, email string) (*User, error) {
	q := fmt.Sprintf("SELECT id, name FROM users WHERE email = '%s'", email)
	row := db.QueryRow(q)
	u := &User{}
	return u, row.Scan(&u.ID, &u.Name)
}''',
         good='''func FindUser(db *sql.DB, email string) (*User, error) {
	row := db.QueryRow("SELECT id, name FROM users WHERE email = $1", email)
	u := &User{}
	return u, row.Scan(&u.ID, &u.Name)
}'''),
    dict(id="go-nil-map", language="go", path="cache/tags.go", category="BUG",
         issue="Writing to a nil map panics: counts is declared but never initialised with make.",
         bad='''func TagCounts(posts []Post) map[string]int {
	var counts map[string]int
	for _, p := range posts {
		for _, t := range p.Tags {
			counts[t]++
		}
	}
	return counts
}''',
         good='''func TagCounts(posts []Post) map[string]int {
	counts := make(map[string]int)
	for _, p := range posts {
		for _, t := range p.Tags {
			counts[t]++
		}
	}
	return counts
}'''),
    dict(id="go-body-leak", language="go", path="client/fetch.go", category="BUG",
         issue="The HTTP response body is never closed, leaking connections; defer resp.Body.Close().",
         bad='''func FetchStatus(url string) (int, error) {
	resp, err := http.Get(url)
	if err != nil {
		return 0, err
	}
	return resp.StatusCode, nil
}''',
         good='''func FetchStatus(url string) (int, error) {
	resp, err := http.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	return resp.StatusCode, nil
}'''),
    dict(id="go-cmd-injection", language="go", path="ops/list.go", category="SECURITY",
         issue="Command injection: the directory name is concatenated into a shell command; call ls directly with an argument.",
         bad='''func ListDir(dir string) ([]byte, error) {
	return exec.Command("sh", "-c", "ls -la "+dir).Output()
}''',
         good='''func ListDir(dir string) ([]byte, error) {
	return exec.Command("ls", "-la", "--", dir).Output()
}'''),
    dict(id="go-concat", language="go", path="report/csv.go", category="PERF",
         issue="String concatenation in a loop is quadratic; use strings.Builder.",
         bad='''func JoinRows(rows []string) string {
	out := ""
	for _, r := range rows {
		out += r + "\\n"
	}
	return out
}''',
         good='''func JoinRows(rows []string) string {
	var b strings.Builder
	for _, r := range rows {
		b.WriteString(r)
		b.WriteByte('\\n')
	}
	return b.String()
}'''),

    # -------------------------------------------------------------------- rust
    dict(id="rs-unwrap", language="rust", path="src/config.rs", category="BUG",
         issue="unwrap on parsing user input panics on bad input; return a Result instead.",
         bad='''pub fn parse_limit(raw: &str) -> u32 {
    raw.trim().parse::<u32>().unwrap()
}''',
         good='''pub fn parse_limit(raw: &str) -> Result<u32, std::num::ParseIntError> {
    raw.trim().parse::<u32>()
}'''),
    dict(id="rs-underflow", language="rust", path="src/page.rs", category="BUG",
         issue="len - 1 underflows (panics in debug, wraps in release) when the slice is empty; handle the empty case.",
         bad='''pub fn last_index(items: &[Item]) -> usize {
    items.len() - 1
}''',
         good='''pub fn last_index(items: &[Item]) -> Option<usize> {
    items.len().checked_sub(1)
}'''),
    dict(id="rs-bounds", language="rust", path="src/diff.rs", category="BUG",
         issue="v[i + 1] goes out of bounds on the last iteration and panics; iterate over windows(2).",
         bad='''pub fn deltas(v: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    for i in 0..v.len() {
        out.push(v[i + 1] - v[i]);
    }
    out
}''',
         good='''pub fn deltas(v: &[i64]) -> Vec<i64> {
    v.windows(2).map(|w| w[1] - w[0]).collect()
}'''),
    dict(id="rs-sqli", language="rust", path="src/db.rs", category="SECURITY",
         issue="SQL injection: the user name is interpolated with format!; use a bound parameter.",
         bad='''pub fn find_user(conn: &Connection, name: &str) -> rusqlite::Result<i64> {
    let sql = format!("SELECT id FROM users WHERE name = '{}'", name);
    conn.query_row(&sql, [], |r| r.get(0))
}''',
         good='''pub fn find_user(conn: &Connection, name: &str) -> rusqlite::Result<i64> {
    conn.query_row("SELECT id FROM users WHERE name = ?1", [name], |r| r.get(0))
}'''),
    dict(id="rs-map-unwrap", language="rust", path="src/prices.rs", category="BUG",
         issue="HashMap::get(..).unwrap() panics for an unknown SKU coming from the request; return an Option or an error.",
         bad='''pub fn price_of(prices: &HashMap<String, u64>, sku: &str) -> u64 {
    *prices.get(sku).unwrap()
}''',
         good='''pub fn price_of(prices: &HashMap<String, u64>, sku: &str) -> Option<u64> {
    prices.get(sku).copied()
}'''),
    dict(id="rs-clone-loop", language="rust", path="src/report.rs", category="PERF",
         issue="data.clone() copies the whole vector just to read it; iterate over a reference.",
         bad='''pub fn total_weight(data: &Vec<Parcel>) -> u64 {
    let mut total = 0;
    for p in data.clone() {
        total += p.weight;
    }
    total
}''',
         good='''pub fn total_weight(data: &[Parcel]) -> u64 {
    data.iter().map(|p| p.weight).sum()
}'''),
    dict(id="rs-secret", language="rust", path="src/client.rs", category="SECURITY",
         issue="An API token is hard-coded in the binary; read it from the environment.",
         bad='''const API_TOKEN: &str = "demo-api-token-0d5a6f8b9c2e7f3a";

pub fn auth_header() -> String {
    format!("Bearer {}", API_TOKEN)
}''',
         good='''pub fn auth_header() -> Result<String, std::env::VarError> {
    Ok(format!("Bearer {}", std::env::var("API_TOKEN")?))
}'''),
    dict(id="rs-float-eq", language="rust", path="src/money.rs", category="BUG",
         issue="Comparing money as f64 with == fails for values like 0.1 + 0.2; use integer minor units (paise/cents).",
         bad='''pub fn is_fully_paid(paid: f64, due: f64) -> bool {
    paid == due
}''',
         good='''pub fn is_fully_paid(paid_paise: u64, due_paise: u64) -> bool {
    paid_paise >= due_paise
}'''),
]


def as_hunk(code: str) -> str:
    """The snippet as an added-lines diff hunk, the shape the reviewer sees."""
    lines = code.splitlines()
    return f"@@ -0,0 +1,{len(lines)} @@\n" + "\n".join("+" + ln for ln in lines)
