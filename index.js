// const dns = require("node:dns");
// dns.setServers(["1.1.1.1", "8.8.8.8"]);
require("dotenv").config({ path: require("path").join(__dirname, ".env") });

const express = require("express");
const cors = require("cors");
const { MongoClient, ObjectId } = require("mongodb");
const { GoogleGenAI } = require("@google/genai");

const app = express();
const port = process.env.PORT || 5000;

const allowedOrigins = [
  process.env.CLIENT_URL,
  "http://localhost:3000",
].filter(Boolean);

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (
      allowedOrigins.includes(origin) ||
      origin.endsWith(".vercel.app") ||
      origin.includes("localhost")
    ) {
      return callback(null, origin);
    }
    return callback(null, origin);
  },
  credentials: true,
}));
app.use(express.json({ limit: "1mb" }));

// ---------- Database ----------
const cleanEnv = (val) => (val || "").split("#")[0].trim().replace(/^["']|["']$/g, "");
const mongoUri = cleanEnv(process.env.MONGO_DB_URI);
const client = new MongoClient(mongoUri);
const db = client.db("loop");
const users = db.collection("user");          // better-auth collection
const sessions = db.collection("session");    // better-auth collection
const orgs = db.collection("organizations");
const members = db.collection("members");     // userId -> organizationId + role
const feedbackCol = db.collection("feedback");
const chatMessagesCol = db.collection("chat_messages");
const reportsCol = db.collection("reports");

// Tenant Collection Aliases
const organization = orgs;
const member = members;
const feedback = feedbackCol;
const chat_messages = chatMessagesCol;

// ---------- Constants ----------
const ROLES = ["org_admin", "manager", "analyst", "viewer"];
const WRITE = ["org_admin", "manager", "analyst"];
const MANAGE = ["org_admin", "manager"];
const ADMIN = ["org_admin"];
const SENTIMENTS = ["positive", "neutral", "negative"];
const STATUSES = ["new", "reviewed", "resolved"];
const RANGES = { "7d": 7, "30d": 30, "90d": 90 };

// ---------- Helpers ----------
const wrap = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch((err) => {
    console.error(err);
    res.status(500).json({ error: "Something went wrong on the server." });
  });

const oid = (id) => (ObjectId.isValid(id) ? new ObjectId(id) : null);
const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- Gemini ----------
const GEMINI_KEY = cleanEnv(process.env.GEMINI_API_KEY);
console.log(GEMINI_KEY ? "Gemini key successfully loaded from process.env." : "WARNING: GEMINI_API_KEY is missing in process.env - AI features will fail.");
// vertexai: false দেওয়া আছে, যাতে কম্পিউটারের GOOGLE_GENAI_USE_VERTEXAI সেটিং এটা বদলে না দেয়
let ai = new GoogleGenAI({ apiKey: GEMINI_KEY, vertexai: false });
let aiMode = "gemini-api";
const MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash";

function aiError(err) {
  const msg = String(err?.message || err);
  const status = Number(err?.status || err?.code) || 0;
  if (/default credentials/i.test(msg))
    return GEMINI_KEY
      ? "Gemini could not authenticate with this key. Create a new API key in Google AI Studio (aistudio.google.com/apikey), put it in the server .env, and fully restart the server."
      : "GEMINI_API_KEY is empty. Add it to D:\\loop server\\.env (same folder as index.js) and fully restart the server.";
  if (/GEMINI_API_KEY|API key must be set|Missing.*api.?key/i.test(msg))
    return "GEMINI_API_KEY is missing. Add it to the server .env file and restart the server.";
  if (status === 401 || /API key not valid|API_KEY_INVALID|UNAUTHENTICATED/i.test(msg))
    return "Gemini rejected the API key. Create a new key in Google AI Studio and update GEMINI_API_KEY in the server .env.";
  if (status === 403 || /PERMISSION_DENIED|leaked/i.test(msg))
    return "Gemini denied access (403). The key may be disabled, for example after it was exposed. Create a new key and update GEMINI_API_KEY.";
  if (status === 404 || /is not found|not supported for generateContent/i.test(msg))
    return `Gemini model "${MODEL}" was not found. Set GEMINI_MODEL=gemini-2.5-flash in the server .env.`;
  if (status === 429 || /RESOURCE_EXHAUSTED|quota/i.test(msg))
    return "Gemini quota or rate limit reached. Wait a minute and try again.";
  return "AI request failed: " + msg.slice(0, 200);
}

async function gemini(prompt, system, json = false) {
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await ai.models.generateContent({
        model: MODEL,
        contents: prompt,
        config: { systemInstruction: system, ...(json ? { responseMimeType: "application/json" } : {}) },
      });
      return res.text || "";
    } catch (err) {
      lastErr = err;
      // "AQ." দিয়ে শুরু হওয়া key Vertex AI express mode-এর হতে পারে, তাই একবার সেই মোডে চেষ্টা করে
      if (/default credentials/i.test(String(err?.message)) && GEMINI_KEY && aiMode === "gemini-api") {
        aiMode = "vertex-express";
        ai = new GoogleGenAI({ apiKey: GEMINI_KEY, vertexai: true });
        console.error("Switching to Vertex AI express mode and retrying...");
        attempt--;
        continue;
      }
      console.error(`Gemini error (attempt ${attempt + 1}):`, err?.status || "", String(err?.message || err).slice(0, 300));
      const status = Number(err?.status || err?.code) || 0;
      if ([400, 401, 403, 404].includes(status)) break; // আবার চেষ্টা করে লাভ নেই
      await sleep(800 * 2 ** attempt);
    }
  }
  throw lastErr;
}
const parseJson = (raw) => JSON.parse(raw.replace(/```json|```/g, "").trim());

const cleanAnalysis = (a) => ({
  sentiment: SENTIMENTS.includes(a?.sentiment) ? a.sentiment : "neutral",
  themes: Array.isArray(a?.themes) ? a.themes.slice(0, 3).map((t) => String(t).toLowerCase().trim()).filter(Boolean) : [],
  summary: typeof a?.summary === "string" ? a.summary.slice(0, 300) : "",
});

const ANALYZE_SYSTEM = `You analyze customer feedback. Reply with JSON only.
For each feedback give: sentiment ("positive"|"neutral"|"negative"), themes (1-3 short lowercase topics such as delivery, pricing, support, quality, app, checkout, refund, ux), summary (one short sentence).`;

async function analyzeOne(text) {
  const raw = await gemini(`Feedback: """${text}"""\nReturn: {"sentiment":"","themes":[],"summary":""}`, ANALYZE_SYSTEM, true);
  return cleanAnalysis(parseJson(raw));
}

async function analyzeBatch(texts) {
  const list = texts.map((t, i) => `${i}. ${t}`).join("\n");
  const raw = await gemini(
    `Feedback list:\n${list}\n\nReturn a JSON array: [{"i":0,"sentiment":"","themes":[],"summary":""}, ...] with one object per feedback.`,
    ANALYZE_SYSTEM,
    true
  );
  const arr = parseJson(raw);
  const out = {};
  for (const item of Array.isArray(arr) ? arr : []) out[item.i] = cleanAnalysis(item);
  return out;
}

async function analyzePending(orgId, limit = 40) {
  const pending = await feedbackCol.find({ organizationId: orgId, aiStatus: { $in: ["pending", "failed"] } }).limit(limit).toArray();
  let done = 0;
  for (let i = 0; i < pending.length; i += 20) {
    const chunk = pending.slice(i, i + 20);
    try {
      const results = await analyzeBatch(chunk.map((f) => f.text));
      for (let idx = 0; idx < chunk.length; idx++) {
        if (!results[idx]) continue;
        await feedbackCol.updateOne({ _id: chunk[idx]._id }, { $set: { ...results[idx], aiStatus: "done" } });
        done++;
      }
    } catch (err) {
      console.error("Batch analysis failed:", err.message);
    }
  }
  return { processed: pending.length, done };
}

// ---------- Auth middleware (better-auth session cookie) ----------
function readSessionToken(req) {
  const header = req.headers.cookie || "";
  for (const part of header.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === "better-auth.session_token" || k === "__Secure-better-auth.session_token") {
      return decodeURIComponent(rest.join("=")).split(".")[0];
    }
  }
  return null;
}

const requireAuth = wrap(async (req, res, next) => {
  const token = readSessionToken(req);
  if (!token) return res.status(401).json({ error: "Please log in." });
  const session = await sessions.findOne({ token });
  if (!session || new Date(session.expiresAt) < new Date()) {
    return res.status(401).json({ error: "Your session has expired. Please log in again." });
  }
  const uid = String(session.userId);
  const user = await users.findOne(ObjectId.isValid(uid) ? { $or: [{ _id: new ObjectId(uid) }, { id: uid }] } : { id: uid });
  if (!user) return res.status(401).json({ error: "User not found." });
  req.user = { id: uid, name: user.name, email: user.email, image: user.image || null };
  next();
});

// organizationId ও role সবসময় এখান থেকে আসে, কখনো request body থেকে নয়
const requireOrg = wrap(async (req, res, next) => {
  const m = await members.findOne({ userId: req.user.id });
  if (!m) return res.status(403).json({ error: "You are not part of an organization yet.", code: "NO_ORG" });
  req.orgId = m.organizationId;
  req.role = m.role;
  next();
});

const can = (...roles) => (req, res, next) =>
  roles.includes(req.role) ? next() : res.status(403).json({ error: "You do not have permission to do this." });

const auth = [requireAuth, requireOrg];

// ---------- Health, Me, Organization ----------
app.get("/api/health", (req, res) => res.json({ ok: true }));

app.get("/api/me", requireAuth, wrap(async (req, res) => {
  const m = await members.findOne({ userId: req.user.id });
  let membership = null;
  if (m) {
    const org = await orgs.findOne({ _id: m.organizationId });
    membership = { organizationId: m.organizationId, organizationName: org?.name || "", role: m.role };
  }
  res.json({ user: req.user, membership });
}));

app.post("/api/organization/onboard", requireAuth, wrap(async (req, res) => {
  const existing = await members.findOne({ userId: req.user.id });
  if (existing) return res.json({ organizationId: existing.organizationId, role: existing.role, existing: true });
  const name = String(req.body.name || "").trim().slice(0, 80) || `${req.user.name}'s organization`;
  const org = await orgs.insertOne({ name, createdBy: req.user.id, createdAt: new Date() });
  try {
    await members.insertOne({ userId: req.user.id, organizationId: org.insertedId, role: "org_admin", createdAt: new Date() });
  } catch (err) {
    await orgs.deleteOne({ _id: org.insertedId });
    const m = await members.findOne({ userId: req.user.id });
    return res.json({ organizationId: m.organizationId, role: m.role, existing: true });
  }
  res.status(201).json({ organizationId: org.insertedId, role: "org_admin", name });
}));

app.get("/api/organization", ...auth, wrap(async (req, res) => {
  const org = await orgs.findOne({ _id: req.orgId });
  const memberCount = await members.countDocuments({ organizationId: req.orgId });
  const feedbackCount = await feedbackCol.countDocuments({ organizationId: req.orgId });
  res.json({ ...org, memberCount, feedbackCount, yourRole: req.role });
}));

app.patch("/api/organization", ...auth, can(...ADMIN), wrap(async (req, res) => {
  const name = String(req.body.name || "").trim().slice(0, 80);
  if (!name) return res.status(400).json({ error: "Name is required." });
  await orgs.updateOne({ _id: req.orgId }, { $set: { name } });
  res.json({ success: true, name });
}));

// ---------- Users / Roles ----------
app.get("/api/users", ...auth, can(...MANAGE), wrap(async (req, res) => {
  const ms = await members.find({ organizationId: req.orgId }).toArray();
  const ids = ms.map((m) => m.userId);
  const objIds = ids.filter(ObjectId.isValid).map((i) => new ObjectId(i));
  const us = await users.find({ $or: [{ _id: { $in: objIds } }, { id: { $in: ids } }] }).toArray();
  const byId = {};
  for (const u of us) byId[String(u._id)] = u;
  res.json(ms.map((m) => ({
    userId: m.userId,
    role: m.role,
    name: byId[m.userId]?.name || "",
    email: byId[m.userId]?.email || "",
    image: byId[m.userId]?.image || null,
    joinedAt: m.createdAt,
  })));
}));

// একজন রেজিস্টার্ড ইউজারকে (যার এখনো কোনো org নেই) এই org-এ যোগ করে
app.post("/api/users", ...auth, can(...ADMIN), wrap(async (req, res) => {
  const email = String(req.body.email || "").trim().toLowerCase();
  const role = req.body.role;
  if (!email || !ROLES.includes(role)) return res.status(400).json({ error: "Valid email and role are required." });
  const user = await users.findOne({ email });
  if (!user) return res.status(404).json({ error: "No account found with this email. Ask them to sign up first." });
  const uid = String(user._id);
  if (await members.findOne({ userId: uid })) return res.status(409).json({ error: "This user already belongs to an organization." });
  await members.insertOne({ userId: uid, organizationId: req.orgId, role, createdAt: new Date() });
  res.status(201).json({ success: true });
}));

app.patch("/api/users/:userId/role", ...auth, can(...ADMIN), wrap(async (req, res) => {
  const { userId } = req.params;
  const role = req.body.role;
  if (!ROLES.includes(role)) return res.status(400).json({ error: "Invalid role." });
  const target = await members.findOne({ userId, organizationId: req.orgId });
  if (!target) return res.status(404).json({ error: "Member not found." });
  if (target.role === "org_admin" && role !== "org_admin") {
    const admins = await members.countDocuments({ organizationId: req.orgId, role: "org_admin" });
    if (admins <= 1) return res.status(400).json({ error: "The organization needs at least one admin." });
  }
  await members.updateOne({ userId, organizationId: req.orgId }, { $set: { role } });
  res.json({ success: true });
}));

app.delete("/api/users/:userId", ...auth, can(...ADMIN), wrap(async (req, res) => {
  const { userId } = req.params;
  if (userId === req.user.id) return res.status(400).json({ error: "You cannot remove yourself." });
  const r = await members.deleteOne({ userId, organizationId: req.orgId });
  if (!r.deletedCount) return res.status(404).json({ error: "Member not found." });
  res.json({ success: true });
}));

// ---------- Feedback ----------
app.post("/api/feedback", ...auth, can(...WRITE), wrap(async (req, res) => {
  const text = String(req.body.text || "").trim();
  if (!text || text.length > 5000) return res.status(400).json({ error: "Feedback text is required (max 5000 characters)." });
  const doc = {
    organizationId: req.orgId,
    text,
    source: String(req.body.source || "manual").slice(0, 30),
    customerName: String(req.body.customerName || "").slice(0, 80),
    status: "new",
    sentiment: null,
    themes: [],
    summary: "",
    aiStatus: "pending",
    createdBy: req.user.id,
    createdAt: new Date(),
  };
  const r = await feedbackCol.insertOne(doc);
  doc._id = r.insertedId;
  // AI ব্যর্থ হলেও feedback সেভ থাকে, পরে আবার চেষ্টা করা যায়
  try {
    const a = await analyzeOne(text);
    await feedbackCol.updateOne({ _id: doc._id }, { $set: { ...a, aiStatus: "done" } });
    Object.assign(doc, a, { aiStatus: "done" });
  } catch (err) {
    console.error("AI analysis failed:", err.message);
    await feedbackCol.updateOne({ _id: doc._id }, { $set: { aiStatus: "failed" } });
    doc.aiStatus = "failed";
  }
  res.status(201).json(doc);
}));

app.post("/api/feedback/analyze-pending", ...auth, can(...WRITE), wrap(async (req, res) => {
  res.json(await analyzePending(req.orgId, 40));
}));

// AI দিয়ে analyze করা অনেকগুলো review একবারে সেভ করে
app.post("/api/feedback/bulk", ...auth, can(...WRITE), wrap(async (req, res) => {
  const items = (Array.isArray(req.body.items) ? req.body.items : []).slice(0, 100);
  const docs = items.map((it) => {
    const text = String(it.text || "").trim();
    if (!text || text.length > 5000) return null;
    return {
      organizationId: req.orgId, text, source: "manual", customerName: "", status: "new",
      ...cleanAnalysis(it), aiStatus: "done", createdBy: req.user.id, createdAt: new Date(),
    };
  }).filter(Boolean);
  if (!docs.length) return res.status(400).json({ error: "No valid reviews to save." });
  await feedbackCol.insertMany(docs);
  res.status(201).json({ inserted: docs.length });
}));

app.get("/api/feedback", ...auth, wrap(async (req, res) => {
  const { search, sentiment, theme, status, source, from, to } = req.query;
  const q = { organizationId: req.orgId };
  if (search) q.text = { $regex: escapeRegex(String(search)), $options: "i" };
  if (SENTIMENTS.includes(sentiment)) q.sentiment = sentiment;
  if (theme) q.themes = String(theme).toLowerCase();
  if (STATUSES.includes(status)) q.status = status;
  if (source) q.source = String(source);
  if (from || to) {
    q.createdAt = {};
    if (from) q.createdAt.$gte = new Date(from);
    if (to) { const d = new Date(to); d.setHours(23, 59, 59, 999); q.createdAt.$lte = d; }
  }
  const limit = Math.min(Math.max(parseInt(req.query.limit) || 20, 1), 100);
  const page = Math.max(parseInt(req.query.page) || 1, 1);
  const total = await feedbackCol.countDocuments(q);
  const items = await feedbackCol.find(q).sort({ createdAt: -1 }).skip((page - 1) * limit).limit(limit).toArray();
  res.json({ items, total, page, pages: Math.ceil(total / limit) || 1 });
}));

app.get("/api/feedback/:id", ...auth, wrap(async (req, res) => {
  const id = oid(req.params.id);
  if (!id) return res.status(400).json({ error: "Invalid id." });
  const doc = await feedbackCol.findOne({ _id: id, organizationId: req.orgId });
  if (!doc) return res.status(404).json({ error: "Feedback not found." });
  res.json(doc);
}));

app.patch("/api/feedback/:id", ...auth, can(...WRITE), wrap(async (req, res) => {
  const id = oid(req.params.id);
  if (!id) return res.status(400).json({ error: "Invalid id." });
  const set = {};
  if (req.body.status !== undefined) {
    if (!STATUSES.includes(req.body.status)) return res.status(400).json({ error: "Invalid status." });
    set.status = req.body.status;
  }
  if (req.body.text !== undefined) {
    const text = String(req.body.text).trim();
    if (!text || text.length > 5000) return res.status(400).json({ error: "Invalid text." });
    set.text = text;
    set.aiStatus = "pending";
  }
  if (!Object.keys(set).length) return res.status(400).json({ error: "Nothing to update." });
  const r = await feedbackCol.updateOne({ _id: id, organizationId: req.orgId }, { $set: set });
  if (!r.matchedCount) return res.status(404).json({ error: "Feedback not found." });
  res.json({ success: true });
}));

app.post("/api/feedback/:id/analyze", ...auth, can(...WRITE), wrap(async (req, res) => {
  const id = oid(req.params.id);
  if (!id) return res.status(400).json({ error: "Invalid id." });
  const doc = await feedbackCol.findOne({ _id: id, organizationId: req.orgId });
  if (!doc) return res.status(404).json({ error: "Feedback not found." });
  try {
    const a = await analyzeOne(doc.text);
    await feedbackCol.updateOne({ _id: id }, { $set: { ...a, aiStatus: "done" } });
    res.json({ ...doc, ...a, aiStatus: "done" });
  } catch (err) {
    console.error(err.message);
    await feedbackCol.updateOne({ _id: id }, { $set: { aiStatus: "failed" } });
    res.status(502).json({ error: aiError(err) });
  }
}));

app.delete("/api/feedback/:id", ...auth, can(...MANAGE), wrap(async (req, res) => {
  const id = oid(req.params.id);
  if (!id) return res.status(400).json({ error: "Invalid id." });
  const r = await feedbackCol.deleteOne({ _id: id, organizationId: req.orgId });
  if (!r.deletedCount) return res.status(404).json({ error: "Feedback not found." });
  res.json({ success: true });
}));

// ---------- Analytics ----------
app.get("/api/analytics", ...auth, wrap(async (req, res) => {
  const match = { organizationId: req.orgId };
  const count = (field) => feedbackCol.aggregate([{ $match: match }, { $group: { _id: `$${field}`, n: { $sum: 1 } } }]).toArray();
  const [total, bySentiment, byStatus, themes, pending] = await Promise.all([
    feedbackCol.countDocuments(match),
    count("sentiment"),
    count("status"),
    feedbackCol.aggregate([
      { $match: match }, { $unwind: "$themes" },
      { $group: { _id: "$themes", count: { $sum: 1 } } },
      { $sort: { count: -1 } }, { $limit: 8 },
    ]).toArray(),
    feedbackCol.countDocuments({ ...match, aiStatus: { $ne: "done" } }),
  ]);
  const sentiment = { positive: 0, neutral: 0, negative: 0 };
  for (const s of bySentiment) if (s._id in sentiment) sentiment[s._id] = s.n;
  const status = { new: 0, reviewed: 0, resolved: 0 };
  for (const s of byStatus) if (s._id in status) status[s._id] = s.n;
  const analyzed = sentiment.positive + sentiment.neutral + sentiment.negative;
  res.json({
    total,
    sentiment,
    status,
    positivePercent: analyzed ? Math.round((sentiment.positive / analyzed) * 100) : 0,
    negativePercent: analyzed ? Math.round((sentiment.negative / analyzed) * 100) : 0,
    topThemes: themes.map((t) => ({ theme: t._id, count: t.count })),
    pendingAnalysis: pending,
  });
}));

app.get("/api/analytics/trends", ...auth, wrap(async (req, res) => {
  const days = RANGES[req.query.range] || 30;
  const since = new Date(); since.setHours(0, 0, 0, 0); since.setDate(since.getDate() - (days - 1));
  const rows = await feedbackCol.aggregate([
    { $match: { organizationId: req.orgId, createdAt: { $gte: since } } },
    { $group: { _id: { d: { $dateToString: { format: "%Y-%m-%d", date: "$createdAt" } }, s: "$sentiment" }, n: { $sum: 1 } } },
  ]).toArray();
  const map = {};
  for (let i = 0; i < days; i++) {
    const d = new Date(since); d.setDate(since.getDate() + i);
    const key = d.toISOString().slice(0, 10);
    map[key] = { date: key, positive: 0, neutral: 0, negative: 0, total: 0 };
  }
  for (const r of rows) {
    const p = map[r._id.d];
    if (!p) continue;
    if (r._id.s in p) p[r._id.s] += r.n;
    p.total += r.n;
  }
  res.json(Object.values(map));
}));

app.get("/api/themes", ...auth, wrap(async (req, res) => {
  const rows = await feedbackCol.aggregate([
    { $match: { organizationId: req.orgId } }, { $unwind: "$themes" },
    { $group: {
        _id: "$themes",
        count: { $sum: 1 },
        positive: { $sum: { $cond: [{ $eq: ["$sentiment", "positive"] }, 1, 0] } },
        neutral: { $sum: { $cond: [{ $eq: ["$sentiment", "neutral"] }, 1, 0] } },
        negative: { $sum: { $cond: [{ $eq: ["$sentiment", "negative"] }, 1, 0] } },
    } },
    { $sort: { count: -1 } },
  ]).toArray();
  res.json(rows.map((r) => ({ theme: r._id, count: r.count, positive: r.positive, neutral: r.neutral, negative: r.negative })));
}));

// ---------- AI ----------
// ব্রাউজারে http://localhost:5000/api/ai/health খুলে Gemini কাজ করছে কিনা দেখা যায় (লগইন থাকতে হবে)
app.get("/api/ai/health", requireAuth, wrap(async (req, res) => {
  try {
    const reply = await gemini("Reply with the single word OK.", "You are a health check.");
    res.json({ ok: true, model: MODEL, mode: aiMode, keyLoaded: !!GEMINI_KEY, reply: reply.trim().slice(0, 20) });
  } catch (err) {
    res.status(502).json({ ok: false, model: MODEL, mode: aiMode, keyLoaded: !!GEMINI_KEY, error: aiError(err) });
  }
}));

app.post("/api/ai/analyze", requireAuth, wrap(async (req, res) => {
  const text = String(req.body.text || "").trim();
  if (!text || text.length > 5000) return res.status(400).json({ error: "Feedback text is required (max 5000 characters)." });
  try { res.json(await analyzeOne(text)); }
  catch (err) { console.error(err.message); res.status(502).json({ error: aiError(err) }); }
}));

// অনেকগুলো review (প্রতি লাইনে একটা) একসাথে analyze করে, কিছু সেভ করে না
app.post("/api/ai/analyze-batch", requireAuth, wrap(async (req, res) => {
  const texts = (Array.isArray(req.body.texts) ? req.body.texts : [])
    .map((t) => String(t).trim()).filter(Boolean).slice(0, 20).map((t) => t.slice(0, 1000));
  if (!texts.length) return res.status(400).json({ error: "Add at least one review." });
  try {
    const results = await analyzeBatch(texts);
    res.json(texts.map((text, i) => ({ text, ...(results[i] || { sentiment: "neutral", themes: [], summary: "" }) })));
  } catch (err) {
    console.error(err.message);
    res.status(502).json({ error: aiError(err) });
  }
}));

app.get("/api/ai/query", ...auth, wrap(async (req, res) => {
  const limit = Math.min(Math.max(parseInt(req.query.limit) || 50, 1), 200);
  const messages = await chatMessagesCol
    .find({ organizationId: req.orgId, userId: req.user.id })
    .sort({ createdAt: 1 })
    .limit(limit)
    .toArray();
  res.json({ messages });
}));

app.post("/api/ai/query", ...auth, wrap(async (req, res) => {
  const question = String(req.body.question || "").trim().slice(0, 8000);
  if (!question) return res.status(400).json({ error: "Question is required." });
  const history = (Array.isArray(req.body.history) ? req.body.history : []).slice(-6)
    .map((h) => `${h.sender === "user" ? "User" : "Assistant"}: ${String(h.text).slice(0, 500)}`).join("\n");

  const userMsg = {
    organizationId: req.orgId,
    userId: req.user.id,
    sender: "user",
    text: question,
    createdAt: new Date(),
  };
  await chatMessagesCol.insertOne(userMsg);

  const match = { organizationId: req.orgId, aiStatus: "done" };
  const [total, recent, bySentiment, themes] = await Promise.all([
    feedbackCol.countDocuments(match),
    feedbackCol.find(match).sort({ createdAt: -1 }).limit(100).project({ text: 1, sentiment: 1, themes: 1, createdAt: 1 }).toArray(),
    feedbackCol.aggregate([{ $match: match }, { $group: { _id: "$sentiment", n: { $sum: 1 } } }]).toArray(),
    feedbackCol.aggregate([{ $match: match }, { $unwind: "$themes" }, { $group: { _id: "$themes", count: { $sum: 1 } } }, { $sort: { count: -1 } }, { $limit: 10 }]).toArray(),
  ]);

  // সংরক্ষিত feedback নেই এবং প্রশ্ন ছোট হলে AI কল না করে সরাসরি জানিয়ে দেয়।
  if (!total && question.length < 200) {
    const fallbackAnswer = "There is no saved feedback yet. Add feedback, or paste some reviews here (or in the Analyze tab) and I will analyze them.";
    const assistantMsg = {
      organizationId: req.orgId,
      userId: req.user.id,
      sender: "assistant",
      text: fallbackAnswer,
      createdAt: new Date(),
    };
    await chatMessagesCol.insertOne(assistantMsg);
    return res.json({ answer: fallbackAnswer, basedOn: 0, total, userMessage: userMsg, assistantMessage: assistantMsg });
  }

  const sentiment = {};
  for (const s of bySentiment) sentiment[s._id] = s.n;
  const list = recent.map((f, i) => `${i + 1}. [${f.createdAt.toISOString().slice(0, 10)}] (${f.sentiment}; ${f.themes.join(", ")}) ${f.text.slice(0, 300)}`).join("\n");
  const dataBlock = total
    ? `Overall (all saved, analyzed feedback): total ${total}; sentiment ${JSON.stringify(sentiment)}; top themes ${JSON.stringify(themes.map((t) => ({ theme: t._id, count: t.count })))}.\n\nMost recent ${recent.length} saved feedback items:\n${list}`
    : "No saved feedback exists yet for this organization.";
  const prompt = `Today: ${new Date().toISOString().slice(0, 10)}
${dataBlock}

${history ? `Conversation so far:\n${history}\n\n` : ""}User message: ${question}`;
  try {
    const answer = await gemini(
      prompt,
      `You are LOOP, an assistant that analyzes customer feedback.
1. If the user's message itself contains pasted customer reviews, analyze those reviews directly: state how many are positive, neutral and negative, list the top themes, summarize the main complaints and praises with short examples, and give 2-3 recommendations.
2. Otherwise answer the question using only the saved data provided. Use the overall stats for totals and percentages, and the recent items for examples and trends. If the data cannot answer it, say so.
Be concise. Use short paragraphs or numbered points and **bold** for key figures.`
    );
    const assistantMsg = {
      organizationId: req.orgId,
      userId: req.user.id,
      sender: "assistant",
      text: answer,
      createdAt: new Date(),
    };
    await chatMessagesCol.insertOne(assistantMsg);
    res.json({ answer, basedOn: recent.length, total, userMessage: userMsg, assistantMessage: assistantMsg });
  } catch (err) {
    console.error(err.message);
    res.status(502).json({ error: aiError(err) });
  }
}));

// ---------- Reports (VoC) ----------
app.post("/api/reports", ...auth, can(...WRITE), wrap(async (req, res) => {
  const days = RANGES[req.body.range] || 30;
  const since = new Date(Date.now() - days * 864e5);
  const match = { organizationId: req.orgId, createdAt: { $gte: since }, aiStatus: "done" };
  const [total, bySentiment, themes, negatives, positives] = await Promise.all([
    feedbackCol.countDocuments(match),
    feedbackCol.aggregate([{ $match: match }, { $group: { _id: "$sentiment", n: { $sum: 1 } } }]).toArray(),
    feedbackCol.aggregate([{ $match: match }, { $unwind: "$themes" }, { $group: { _id: "$themes", count: { $sum: 1 } } }, { $sort: { count: -1 } }, { $limit: 8 }]).toArray(),
    feedbackCol.find({ ...match, sentiment: "negative" }).sort({ createdAt: -1 }).limit(15).toArray(),
    feedbackCol.find({ ...match, sentiment: "positive" }).sort({ createdAt: -1 }).limit(10).toArray(),
  ]);
  if (!total) return res.status(400).json({ error: "No analyzed feedback found in this period." });
  const sentiment = { positive: 0, neutral: 0, negative: 0 };
  for (const s of bySentiment) if (s._id in sentiment) sentiment[s._id] = s.n;
  const topThemes = themes.map((t) => ({ theme: t._id, count: t.count }));

  const prompt = `Period: last ${days} days. Total feedback: ${total}. Sentiment: ${JSON.stringify(sentiment)}. Top themes: ${JSON.stringify(topThemes)}.
Negative examples:\n${negatives.map((f) => `- ${f.text.slice(0, 250)}`).join("\n")}
Positive examples:\n${positives.map((f) => `- ${f.text.slice(0, 250)}`).join("\n")}
Write a Voice of Customer report as JSON: {"title":"","executiveSummary":"2-3 sentences","topComplaints":[{"theme":"","detail":"","exampleQuote":""}],"topPraises":[{"theme":"","detail":"","exampleQuote":""}],"recommendations":["3-5 concrete actions"]}`;
  let content;
  try {
    content = parseJson(await gemini(prompt, "You are a customer insights analyst. Use only the data given. Reply with JSON only.", true));
  } catch (err) {
    console.error(err.message);
    return res.status(502).json({ error: aiError(err) });
  }
  const doc = {
    organizationId: req.orgId, type: "voc", range: `${days}d`,
    period: { from: since, to: new Date() },
    stats: { total, sentiment, topThemes },
    content, createdBy: req.user.id, createdAt: new Date(),
  };
  const r = await reportsCol.insertOne(doc);
  res.status(201).json({ ...doc, _id: r.insertedId });
}));

app.get("/api/reports", ...auth, wrap(async (req, res) => {
  const list = await reportsCol.find({ organizationId: req.orgId }).sort({ createdAt: -1 })
    .project({ type: 1, range: 1, stats: 1, "content.title": 1, createdAt: 1 }).toArray();
  res.json(list);
}));

app.get("/api/reports/:id", ...auth, wrap(async (req, res) => {
  const id = oid(req.params.id);
  if (!id) return res.status(400).json({ error: "Invalid id." });
  const doc = await reportsCol.findOne({ _id: id, organizationId: req.orgId });
  if (!doc) return res.status(404).json({ error: "Report not found." });
  res.json(doc);
}));

app.delete("/api/reports/:id", ...auth, can(...MANAGE), wrap(async (req, res) => {
  const id = oid(req.params.id);
  if (!id) return res.status(400).json({ error: "Invalid id." });
  const r = await reportsCol.deleteOne({ _id: id, organizationId: req.orgId });
  if (!r.deletedCount) return res.status(404).json({ error: "Report not found." });
  res.json({ success: true });
}));

// ---------- Demo data (শুধু org_admin) ----------
const SAMPLES = [
  "My order arrived three days late and nobody told me.", "Fast delivery, the package arrived a day early!",
  "Support solved my problem in five minutes. Great service!", "I waited two days for a reply from customer support.",
  "Too expensive compared to other stores.", "Great value for the price, I will buy again.",
  "The app keeps crashing when I open my cart.", "The new app design is clean and easy to use.",
  "Product quality is excellent and feels premium.", "The item broke after one week of use.",
  "Payment failed twice before it went through.", "Checkout was smooth and quick.",
  "Refund took more than two weeks to arrive.", "Return process was simple and quick.",
  "The product is okay, it does what it says.", "Packaging was standard, nothing special.",
  "Delivery guy was rude and left the parcel outside.", "Love the discounts and loyalty points.",
  "The website is slow on my phone.", "Customer support agent was very polite and helpful.",
  "Prices went up again without any notice.", "Size chart was wrong, the shirt did not fit.",
  "Amazing quality, exactly as described.", "Tracking page never updates, very frustrating.",
];
const SOURCES = ["email", "survey", "app_store", "social", "support_ticket"];

app.post("/api/dev/seed", ...auth, can(...ADMIN), wrap(async (req, res) => {
  const now = Date.now();
  const docs = Array.from({ length: 48 }, (_, i) => ({
    organizationId: req.orgId,
    text: SAMPLES[i % SAMPLES.length],
    source: SOURCES[Math.floor(Math.random() * SOURCES.length)],
    customerName: "",
    status: "new", sentiment: null, themes: [], summary: "", aiStatus: "pending",
    createdBy: req.user.id,
    createdAt: new Date(now - Math.floor(Math.random() * 60 * 864e5)),
  }));
  await feedbackCol.insertMany(docs);
  const result = await analyzePending(req.orgId, 48);
  res.json({ inserted: docs.length, ...result });
}));

// ---------- Start ----------
app.use((req, res) => res.status(404).json({ error: "Route not found." }));

(async () => {
  await client.connect();
  await members.createIndex({ userId: 1 }, { unique: true });
  await members.createIndex({ organizationId: 1 });
  await orgs.createIndex({ _id: 1, organizationId: 1 }, { sparse: true });
  await feedbackCol.createIndex({ organizationId: 1, createdAt: -1 });
  await chatMessagesCol.createIndex({ organizationId: 1, userId: 1, createdAt: 1 });
  app.listen(port, () => console.log(`LOOP server running on port ${port}`));
})().catch((err) => { console.error("Failed to start:", err); process.exit(1); });

module.exports = app;