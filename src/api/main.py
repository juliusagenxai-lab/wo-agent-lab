"""
FastAPI application — WUApp WriteUp Learning System.

Endpoints:
  GET  /          → serves the unified HTML app
  POST /score     → score a single essay → AssessmentScore
  POST /score/all → score all three genres → StudentReport

Run:
  python -m hypercorn src.api.main:app --bind 0.0.0.0:8000 --reload
"""
import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="WUApp — WriteUp Learning System",
    description="RAG-powered CCSS writing assessment for Grades 6–8",
    version="1.0.0",
)


# ── Request / response models ─────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    grade: int = Field(..., ge=6, le=8, description="Grade level (6, 7, or 8)")
    genre: Literal["argumentative", "informative", "narrative"]
    essay_text: str = Field(..., min_length=1, max_length=5000)


class ScoreAllRequest(BaseModel):
    grade: int = Field(..., ge=6, le=8)
    argumentative: str = Field(..., min_length=1, max_length=5000)
    informative: str = Field(..., min_length=1, max_length=5000)
    narrative: str = Field(..., min_length=1, max_length=5000)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/score")
async def score_single(req: ScoreRequest):
    """Score a single student essay and return an AssessmentScore."""
    from src.scoring.scorer import score_essay

    t0 = time.perf_counter()
    logger.info("POST /score grade=%d genre=%s chars=%d", req.grade, req.genre, len(req.essay_text))

    try:
        result = score_essay(
            grade=req.grade,
            genre=req.genre,
            essay_text=req.essay_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - t0
    logger.info("POST /score completed in %.2fs level=%s", elapsed, result.performance_level)
    return result.model_dump()


@app.post("/score/all")
async def score_all(req: ScoreAllRequest):
    """Score all three genre essays and return a StudentReport."""
    from src.scoring.scorer import score_student

    t0 = time.perf_counter()
    logger.info("POST /score/all grade=%d", req.grade)

    try:
        report = score_student(
            grade=req.grade,
            essays={
                "argumentative": req.argumentative,
                "informative": req.informative,
                "narrative": req.narrative,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - t0
    logger.info("POST /score/all completed in %.2fs level=%s", elapsed, report.overall_level)
    return report.model_dump()


# ── HTML frontend ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WUApp — WriteUp Learning System</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/lucide@latest"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  body { font-family: 'Inter', sans-serif; background-color: #121624; }
  .sidebar-bg { background-color: #161c33; }
  .sidebar-active { background-color: #374279; color: #ffffff; }
  .tab-active { border-bottom: 2px solid #2563eb; color: #1e3a8a; font-weight: 600; }
  .tab-inactive { color: #64748b; }
  .tab-inactive:hover { color: #334155; background-color: #f1f5f9; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
  ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
  .spinner {
    border: 4px solid rgba(37,99,235,0.15);
    width: 48px; height: 48px;
    border-radius: 50%;
    border-left-color: #2563eb;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .bar-fill { transition: width 1s ease-out; }
</style>
</head>
<body class="bg-slate-900 text-slate-100 antialiased min-h-screen flex flex-col">

<!-- Top toolbar -->
<header class="h-12 bg-[#1e2336] border-b border-slate-700/60 px-4 flex items-center justify-between text-slate-300 text-sm select-none shrink-0">
  <div class="flex items-center gap-3">
    <span class="font-medium text-slate-200">WriteUp Learning Management System Dashboard</span>
  </div>
  <div class="flex items-center bg-[#121624] p-1 rounded-lg border border-slate-700/50">
    <span class="px-3 py-1 rounded-md text-xs font-semibold bg-slate-700 text-white shadow-sm">Preview</span>
  </div>
  <div class="flex items-center gap-2 text-slate-400 text-xs">
    <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse inline-block"></span>
    <span class="text-slate-300">RAG Scorer: Live</span>
  </div>
</header>

<div class="flex-1 flex overflow-hidden">
  <!-- Sidebar -->
  <aside class="w-64 sidebar-bg text-slate-300 flex flex-col min-h-full shrink-0">
    <div class="p-6 pb-4 border-b border-slate-700/40">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-indigo-600 text-white rounded-xl shadow-md">
          <i data-lucide="graduation-cap" class="w-6 h-6"></i>
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white leading-none">WUApp</h1>
          <p class="text-[10px] font-semibold tracking-wider text-slate-400 uppercase mt-1">WriteUp Learning System</p>
        </div>
      </div>
    </div>
    <nav class="flex-1 px-4 py-6 space-y-6">
      <div>
        <p class="px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Main Menu</p>
        <div class="space-y-1">
          <button id="nav-dashboard" onclick="switchTab('dashboard')" class="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium sidebar-active transition-colors">
            <i data-lucide="gauge" class="w-5 h-5"></i><span>Dashboard</span>
          </button>
          <button id="nav-grade-level" onclick="switchTab('grade-level')" class="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium hover:bg-slate-800/60 hover:text-white transition-colors">
            <i data-lucide="bookmark" class="w-5 h-5"></i><span>Grade Level</span>
          </button>
        </div>
      </div>
      <div>
        <p class="px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Insights & System</p>
        <div class="space-y-1">
          <button id="nav-analytics" onclick="switchTab('analytics')" class="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium hover:bg-slate-800/60 hover:text-white transition-colors">
            <i data-lucide="bar-chart-3" class="w-5 h-5"></i><span>Analytics</span>
          </button>
        </div>
      </div>
    </nav>
  </aside>

  <!-- Main content -->
  <main class="flex-1 flex flex-col bg-slate-50 text-slate-800 overflow-y-auto">
    <header class="h-16 bg-white border-b border-slate-200 px-8 flex items-center justify-between shrink-0">
      <h2 id="header-title" class="text-lg font-bold text-slate-900">Dashboard</h2>
    </header>

    <div class="p-8 flex-1">
      <!-- DASHBOARD VIEW -->
      <section id="view-dashboard" class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-slate-900 tracking-tight">Welcome, Emerging Writer 👋</h1>
          <p class="text-slate-500 mt-1 text-sm">Select your grade, choose a writing activity, and get instant AI-powered feedback scored against CCSS rubrics.</p>
        </div>
        <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-6">
          <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-100 pb-5">
            <div>
              <h3 class="text-lg font-bold text-slate-800">Writing Activities by Grade Level</h3>
              <p class="text-xs text-slate-500 mt-0.5">Click any card to open the writing studio and score your essay.</p>
            </div>
            <div class="flex items-center gap-2 bg-slate-100 p-1 rounded-lg">
              <label class="cursor-pointer px-3 py-1.5 rounded-md text-xs font-semibold transition-all bg-white text-indigo-600 shadow-sm" id="label-grade-6">
                <input type="radio" name="grade_selector" value="6" checked onchange="selectGrade(6)" class="sr-only"> Grade 6
              </label>
              <label class="cursor-pointer px-3 py-1.5 rounded-md text-xs font-semibold transition-all text-slate-600 hover:text-slate-900" id="label-grade-7">
                <input type="radio" name="grade_selector" value="7" onchange="selectGrade(7)" class="sr-only"> Grade 7
              </label>
              <label class="cursor-pointer px-3 py-1.5 rounded-md text-xs font-semibold transition-all text-slate-600 hover:text-slate-900" id="label-grade-8">
                <input type="radio" name="grade_selector" value="8" onchange="selectGrade(8)" class="sr-only"> Grade 8
              </label>
            </div>
          </div>
          <div id="tasks-container" class="space-y-6"></div>
        </div>
      </section>

      <!-- GRADE LEVEL VIEW -->
      <section id="view-grade-level" class="space-y-6 hidden">
        <div>
          <h1 class="text-2xl font-bold text-slate-900 tracking-tight">Grade Level Management</h1>
          <p class="text-slate-500 mt-1 text-sm">Overview of grade levels and academic writing targets.</p>
        </div>
        <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <table class="w-full text-left border-collapse">
            <thead><tr class="bg-slate-50 border-b border-slate-200 text-xs font-semibold uppercase text-slate-500">
              <th class="py-3.5 px-6">Grade</th><th class="py-3.5 px-6">Focus Area</th>
              <th class="py-3.5 px-6">Target Level</th><th class="py-3.5 px-6">Learners</th><th class="py-3.5 px-6">Status</th>
            </tr></thead>
            <tbody class="divide-y divide-slate-200 text-sm text-slate-700">
              <tr class="hover:bg-slate-50/80"><td class="py-4 px-6 font-medium">Grade 6</td><td class="py-4 px-6">School, House, Neighbourhood &amp; Daily Life</td><td class="py-4 px-6">Foundational</td><td class="py-4 px-6">40</td><td class="py-4 px-6"><span class="px-2.5 py-1 text-xs font-medium rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Active</span></td></tr>
              <tr class="hover:bg-slate-50/80"><td class="py-4 px-6 font-medium">Grade 7</td><td class="py-4 px-6">Foundational Paragraphs &amp; Community</td><td class="py-4 px-6">Developing</td><td class="py-4 px-6">42</td><td class="py-4 px-6"><span class="px-2.5 py-1 text-xs font-medium rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Active</span></td></tr>
              <tr class="hover:bg-slate-50/80"><td class="py-4 px-6 font-medium">Grade 8</td><td class="py-4 px-6">Expository, Descriptive &amp; Argumentative</td><td class="py-4 px-6">Proficient</td><td class="py-4 px-6">38</td><td class="py-4 px-6"><span class="px-2.5 py-1 text-xs font-medium rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Active</span></td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- ANALYTICS VIEW -->
      <section id="view-analytics" class="space-y-6 hidden">
        <div>
          <h1 class="text-2xl font-bold text-slate-900 tracking-tight">System Analytics</h1>
          <p class="text-slate-500 mt-1 text-sm">Performance and summary metrics.</p>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-center gap-5">
            <div class="p-4 bg-indigo-50 rounded-xl text-indigo-600"><i data-lucide="bookmark" class="w-8 h-8"></i></div>
            <div><p class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Grade Levels Supported</p><h3 class="text-3xl font-bold text-slate-900 mt-1">3</h3></div>
          </div>
          <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-center gap-5">
            <div class="p-4 bg-emerald-50 rounded-xl text-emerald-600"><i data-lucide="users" class="w-8 h-8"></i></div>
            <div><p class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Active Learners</p><h3 class="text-3xl font-bold text-slate-900 mt-1">120</h3></div>
          </div>
        </div>
      </section>
    </div>
  </main>
</div>

<!-- ═══════════════════════════════════════════════════════════
     WRITING STUDIO MODAL
     ═══════════════════════════════════════════════════════════ -->
<div id="studio-modal" class="fixed inset-0 bg-slate-900/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
  <div class="bg-white rounded-2xl shadow-2xl w-full max-w-5xl flex flex-col overflow-hidden" style="max-height:92vh;">

    <!-- Modal header -->
    <div class="bg-slate-900 text-white px-6 py-4 flex items-center justify-between shrink-0">
      <div class="flex items-center gap-3">
        <div class="p-1.5 bg-indigo-600 rounded-lg"><i data-lucide="pen-line" class="w-4 h-4"></i></div>
        <div>
          <p id="modal-genre-tag" class="text-[11px] font-bold uppercase tracking-wider text-indigo-300"></p>
          <h3 id="modal-prompt-title" class="text-base font-bold leading-tight"></h3>
        </div>
      </div>
      <button onclick="closeStudio()" class="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors">
        <i data-lucide="x" class="w-5 h-5"></i>
      </button>
    </div>

    <!-- Modal body — split pane -->
    <div class="flex flex-1 overflow-hidden min-h-0">

      <!-- Left: essay editor -->
      <div class="w-2/5 border-r border-slate-200 flex flex-col bg-white shrink-0 overflow-y-auto">
        <!-- scrollable content -->
        <div class="flex flex-col gap-4 p-5 flex-1">
          <div>
            <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">Your Essay</label>
            <textarea id="studio-essay" rows="10"
              class="w-full p-3 border border-slate-300 rounded-xl text-sm text-slate-700 leading-relaxed resize-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-shadow"
              placeholder="Write or paste your essay here…"></textarea>
            <p class="text-xs text-slate-400 mt-1 text-right"><span id="char-count">0</span> / 5000 characters</p>
          </div>

          <div class="bg-slate-50 rounded-xl p-4 border border-slate-200 space-y-3">
            <p class="text-xs font-bold text-slate-500 uppercase tracking-wider">Settings</p>
            <div>
              <label class="block text-xs font-semibold text-slate-600 mb-1">Grade Level</label>
              <select id="studio-grade" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-700 focus:ring-2 focus:ring-indigo-500">
                <option value="6">Grade 6</option>
                <option value="7">Grade 7</option>
                <option value="8">Grade 8</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-semibold text-slate-600 mb-1">Writing Genre</label>
              <select id="studio-genre" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-700 focus:ring-2 focus:ring-indigo-500">
                <option value="argumentative">Argumentative</option>
                <option value="informative">Informative</option>
                <option value="narrative">Narrative</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Score button — sticky so it's always visible -->
        <div class="sticky bottom-0 bg-white border-t border-slate-200 p-4">
          <button onclick="scoreEssay()" class="w-full bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white font-bold py-3 px-4 rounded-xl shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2">
            <i data-lucide="zap" class="w-4 h-4"></i> Score Essay
          </button>
        </div>
      </div>

      <!-- Right: results panel -->
      <div class="flex-1 flex flex-col overflow-hidden bg-slate-50 relative">

        <!-- Empty state -->
        <div id="res-empty" class="absolute inset-0 flex flex-col items-center justify-center text-center p-8">
          <div class="w-20 h-20 bg-indigo-50 rounded-full flex items-center justify-center mb-4">
            <i data-lucide="file-text" class="w-10 h-10 text-indigo-300"></i>
          </div>
          <h3 class="text-xl font-bold text-slate-700 mb-1">Ready to Score</h3>
          <p class="text-slate-400 text-sm max-w-xs">Write or paste your essay on the left, then click <strong>Score Essay</strong> to get instant CCSS rubric feedback.</p>
        </div>

        <!-- Loading state -->
        <div id="res-loading" class="hidden absolute inset-0 flex flex-col items-center justify-center bg-white/90 backdrop-blur-sm z-10">
          <div class="spinner mb-5"></div>
          <h3 class="text-lg font-bold text-blue-800 mb-1">Scoring your essay…</h3>
          <p id="loading-msg" class="text-slate-500 text-sm">Retrieving rubric criteria…</p>
        </div>

        <!-- Results -->
        <div id="res-results" class="hidden flex-col h-full">

          <!-- Tabs -->
          <div class="bg-white border-b border-slate-200 px-5 pt-3 flex gap-1 shrink-0">
            <button onclick="showResultTab('review')" id="rtab-review" class="px-4 py-2.5 text-sm rounded-t-lg tab-active transition-colors">
              <i data-lucide="clipboard-list" class="w-3.5 h-3.5 inline-block mr-1 mb-0.5"></i>Score Report
            </button>
            <button onclick="showResultTab('feedback')" id="rtab-feedback" class="px-4 py-2.5 text-sm rounded-t-lg tab-inactive transition-colors">
              <i data-lucide="message-square" class="w-3.5 h-3.5 inline-block mr-1 mb-0.5"></i>Tutor Feedback
            </button>
          </div>

          <!-- Tab panels -->
          <div class="flex-1 overflow-y-auto p-5 space-y-5">

            <!-- Score Report tab -->
            <div id="rtab-panel-review">

              <!-- Summary banner -->
              <div id="score-banner" class="rounded-xl p-4 flex items-center justify-between mb-4 border"></div>

              <!-- Criterion bars -->
              <div id="criteria-bars" class="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4"></div>
            </div>

            <!-- Tutor Feedback tab -->
            <div id="rtab-panel-feedback" class="hidden space-y-4">
              <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5 border-l-4 border-l-green-500">
                <h4 class="text-sm font-bold text-green-700 mb-2 flex items-center gap-1.5">
                  <i data-lucide="thumbs-up" class="w-4 h-4"></i> Strengths to Celebrate
                </h4>
                <ul id="fb-strengths" class="list-disc list-inside text-sm text-slate-700 space-y-1"></ul>
              </div>
              <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5 border-l-4 border-l-amber-500">
                <h4 class="text-sm font-bold text-amber-700 mb-2 flex items-center gap-1.5">
                  <i data-lucide="trending-up" class="w-4 h-4"></i> Next Steps to Improve
                </h4>
                <ul id="fb-steps" class="space-y-2 text-sm text-slate-700"></ul>
              </div>
              <div class="bg-white rounded-xl border border-slate-200 shadow-sm p-5 border-l-4 border-l-purple-500">
                <h4 class="text-sm font-bold text-purple-700 mb-2 flex items-center gap-1.5">
                  <i data-lucide="help-circle" class="w-4 h-4"></i> Questions to Guide Your Revision
                </h4>
                <ul id="fb-questions" class="list-disc list-inside text-sm text-slate-700 italic space-y-1"></ul>
              </div>
            </div>

          </div>
        </div><!-- /res-results -->
      </div><!-- /right pane -->
    </div><!-- /split pane -->
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════
     JAVASCRIPT
     ═══════════════════════════════════════════════════════════ -->
<script>
lucide.createIcons();

// ── Curriculum cards — flat, sorted by unit number ──────────
// Each entry: { unitNum, unit, genre, prompt }
const activities = {
  6: [
    { unitNum:  1, unit: "Unit 1 — My School",        genre: "informative",   prompt: "Write a paragraph describing your school" },
    { unitNum:  3, unit: "Unit 3 — My Best Friend",   genre: "narrative",     prompt: "Write a diary entry about your best friend" },
    { unitNum:  4, unit: "Unit 4 — My Neighbourhood", genre: "informative",   prompt: "Write a paragraph about your neighbourhood" },
    { unitNum:  6, unit: "Unit 6 — Tet Holiday",      genre: "narrative",     prompt: "Write about your favorite holiday memory" },
    { unitNum:  7, unit: "Unit 7 — Television",        genre: "argumentative", prompt: "Write a paragraph about limiting screen time" },
    { unitNum: 10, unit: "Unit 10 — Future Houses",   genre: "informative",   prompt: "Write a paragraph about your dream house" },
    { unitNum: 11, unit: "Unit 11 — Environment",     genre: "argumentative", prompt: "Write a paragraph about improving the environment" },
  ],
  7: [
    { unitNum:  1, unit: "Unit 1 — Hobbies",             genre: "narrative",     prompt: "Write a paragraph about your hobby" },
    { unitNum:  2, unit: "Unit 2 — Healthy Living",      genre: "informative",   prompt: "Write a passage giving advice on staying healthy" },
    { unitNum:  3, unit: "Unit 3 — Community Service",   genre: "narrative",     prompt: "Write about a community service experience" },
    { unitNum:  4, unit: "Unit 4 — Music and Arts",      genre: "narrative",     prompt: "Write an informal letter of invitation to a music event" },
    { unitNum:  5, unit: "Unit 5 — Food and Drink",      genre: "informative",   prompt: "Write a paragraph about healthy eating habits" },
    { unitNum:  7, unit: "Unit 7 — Traffic",             genre: "argumentative", prompt: "Write a paragraph about traffic problems and solutions" },
    { unitNum:  9, unit: "Unit 9 — Festivals",           genre: "narrative",     prompt: "Write about a festival your family celebrates" },
    { unitNum: 10, unit: "Unit 10 — Energy Sources",     genre: "argumentative", prompt: "Write a paragraph about how you save energy at home" },
    { unitNum: 11, unit: "Unit 11 — Future Transport",   genre: "argumentative", prompt: "Write a paragraph about the advantages of a transport type" },
  ],
  8: [
    { unitNum:  1, unit: "Unit 1 — Leisure Time",           genre: "narrative",     prompt: "Write about how you spend your free time" },
    { unitNum:  2, unit: "Unit 2 — Countryside Life",       genre: "informative",   prompt: "Write about the advantages and disadvantages of countryside life" },
    { unitNum:  3, unit: "Unit 3 — Teenagers",              genre: "argumentative", prompt: "Write about the causes of teen stress and solutions" },
    { unitNum:  4, unit: "Unit 4 — Ethnic Groups",          genre: "narrative",     prompt: "Write a paragraph about helping your family" },
    { unitNum:  5, unit: "Unit 5 — Customs & Traditions",   genre: "narrative",     prompt: "Write about an important family tradition" },
    { unitNum:  6, unit: "Unit 6 — Lifestyles",             genre: "informative",   prompt: "Write about the advantages or disadvantages of online learning" },
    { unitNum:  7, unit: "Unit 7 — Environmental Protection", genre: "informative", prompt: "Write a notice about a school environmental event" },
    { unitNum:  8, unit: "Unit 8 — Shopping",               genre: "informative",   prompt: "Write about the advantages or disadvantages of online shopping" },
    { unitNum:  9, unit: "Unit 9 — Natural Disasters",      genre: "informative",   prompt: "Write instructions for before, during, and after a flood" },
    { unitNum: 10, unit: "Unit 10 — Communication",         genre: "informative",   prompt: "Write a paragraph about a modern communication tool" },
    { unitNum: 11, unit: "Unit 11 — Science & Technology",  genre: "argumentative", prompt: "Do you agree that robots will replace teachers? Argue your position" },
    { unitNum: 12, unit: "Unit 12 — Life on Other Planets", genre: "narrative",     prompt: "Write a paragraph imagining life on another planet" },
  ]
};

let currentGrade = 6;

function selectGrade(grade) {
  currentGrade = grade;
  [6,7,8].forEach(g => {
    const lbl = document.getElementById(`label-grade-${g}`);
    if (g === grade) {
      lbl.className = "cursor-pointer px-3 py-1.5 rounded-md text-xs font-semibold transition-all bg-white text-indigo-600 shadow-sm";
    } else {
      lbl.className = "cursor-pointer px-3 py-1.5 rounded-md text-xs font-semibold transition-all text-slate-600 hover:text-slate-900";
    }
  });
  renderCards(grade);
}

function renderCards(grade) {
  const container = document.getElementById('tasks-container');
  const genreColors = { argumentative: 'indigo', informative: 'emerald', narrative: 'amber' };
  const genreLabels = { argumentative: 'Argumentative', informative: 'Informative', narrative: 'Narrative' };

  let html = '<div class="grid grid-cols-1 md:grid-cols-2 gap-3">';
  activities[grade].forEach(task => {
    const col        = genreColors[task.genre] || 'slate';
    const safePrompt = task.prompt.replace(/"/g, '&quot;');
    const safeUnit   = task.unit.replace(/"/g, '&quot;');
    html += `
      <div onclick='openStudio(${grade}, "${task.genre}", "${safeUnit}", "${safePrompt}")'
           class="p-4 rounded-xl border border-slate-200 bg-slate-50/50 hover:bg-${col}-50/40 hover:border-${col}-300 transition-all cursor-pointer group flex flex-col justify-between gap-2">
        <div>
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs font-bold text-slate-600 group-hover:text-${col}-700 transition-colors">${task.unit}</span>
            <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-${col}-100 text-${col}-700">${genreLabels[task.genre]}</span>
          </div>
          <p class="text-sm font-semibold text-slate-800">${task.prompt}</p>
        </div>
        <div class="flex items-center gap-1.5 text-[11px] font-semibold text-${col}-600 pt-1">
          <i data-lucide="pen-line" class="w-3.5 h-3.5"></i>
          <span>Write &amp; Score</span>
        </div>
      </div>`;
  });
  html += '</div>';
  container.innerHTML = html;
  lucide.createIcons();
}

// ── Studio modal ─────────────────────────────────────────────
function openStudio(grade, genre, unit, prompt) {
  document.getElementById('modal-genre-tag').textContent = `Grade ${grade} · ${genre}`;
  document.getElementById('modal-prompt-title').textContent = prompt;
  document.getElementById('studio-essay').value = '';
  document.getElementById('char-count').textContent = '0';
  document.getElementById('studio-grade').value = String(grade);
  document.getElementById('studio-genre').value = genre;

  // Reset results panel
  showState('empty');
  document.getElementById('studio-modal').classList.remove('hidden');
  lucide.createIcons();
  setTimeout(() => document.getElementById('studio-essay').focus(), 100);
}

function closeStudio() {
  document.getElementById('studio-modal').classList.add('hidden');
}

document.getElementById('studio-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('studio-modal')) closeStudio();
});

document.getElementById('studio-essay').addEventListener('input', function() {
  document.getElementById('char-count').textContent = this.value.length;
});

// ── Scoring ──────────────────────────────────────────────────
const loadingMessages = [
  'Retrieving rubric criteria…',
  'Building scoring prompt…',
  'Invoking AI scorer…',
  'Parsing results…'
];

async function scoreEssay() {
  const essay = document.getElementById('studio-essay').value.trim();
  const grade = parseInt(document.getElementById('studio-grade').value);
  const genre = document.getElementById('studio-genre').value;

  if (!essay) {
    document.getElementById('studio-essay').focus();
    document.getElementById('studio-essay').classList.add('ring-2','ring-red-400');
    setTimeout(() => document.getElementById('studio-essay').classList.remove('ring-2','ring-red-400'), 1500);
    return;
  }

  showState('loading');

  // Cycle loading messages
  let msgIdx = 0;
  const msgInterval = setInterval(() => {
    document.getElementById('loading-msg').textContent = loadingMessages[msgIdx % loadingMessages.length];
    msgIdx++;
  }, 1200);

  try {
    const res = await fetch('/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grade, genre, essay_text: essay })
    });

    clearInterval(msgInterval);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
    showState('results');
    showResultTab('review');

  } catch (err) {
    clearInterval(msgInterval);
    showState('empty');
    alert(`Scoring failed: ${err.message}`);
  }
}

// ── Render results ───────────────────────────────────────────
const LEVEL_LABELS = { 1: 'Beginning', 2: 'Developing', 3: 'Proficient', 4: 'Advanced' };
const BAND_COLORS  = {
  Advanced:   { bg: 'bg-emerald-50',  border: 'border-emerald-200', text: 'text-emerald-700', bar: 'bg-emerald-500' },
  Proficient: { bg: 'bg-blue-50',     border: 'border-blue-200',    text: 'text-blue-700',    bar: 'bg-blue-500'    },
  Developing: { bg: 'bg-amber-50',    border: 'border-amber-200',   text: 'text-amber-700',   bar: 'bg-amber-400'   },
  Beginning:  { bg: 'bg-red-50',      border: 'border-red-200',     text: 'text-red-700',     bar: 'bg-red-400'     },
};

function renderResults(data) {
  const level  = data.performance_level;
  const colors = BAND_COLORS[level] || BAND_COLORS.Beginning;

  // ── Summary banner
  document.getElementById('score-banner').className =
    `rounded-xl p-4 flex items-center justify-between mb-4 border ${colors.bg} ${colors.border}`;
  document.getElementById('score-banner').innerHTML = `
    <div>
      <p class="text-xs font-bold uppercase tracking-wider ${colors.text} mb-0.5">
        Grade ${data.grade} · ${data.genre} · ${data.standard}
      </p>
      <p class="text-2xl font-bold text-slate-900">${data.total} <span class="text-base font-medium text-slate-500">/ 24 points</span></p>
    </div>
    <div class="text-right">
      <span class="inline-block px-3 py-1.5 rounded-full text-sm font-bold ${colors.bg} ${colors.text} border ${colors.border}">${level}</span>
      <p class="text-xs text-slate-500 mt-1">${data.criteria.length} criteria scored</p>
    </div>`;

  // ── Criterion bars
  const barsEl = document.getElementById('criteria-bars');
  barsEl.innerHTML = `<h4 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">Criterion Breakdown</h4>`;
  data.criteria.forEach(c => {
    const pct  = (c.level / 4) * 100;
    const col  = BAND_COLORS[LEVEL_LABELS[c.level]] || BAND_COLORS.Beginning;
    barsEl.innerHTML += `
      <div class="pb-3 border-b border-slate-100 last:border-0 last:pb-0">
        <div class="flex items-center justify-between mb-1">
          <span class="text-sm font-semibold text-slate-700">${c.criterion_name}</span>
          <span class="text-xs font-bold px-2 py-0.5 rounded-full ${col.bg} ${col.text} border ${col.border}">${LEVEL_LABELS[c.level]} (${c.level}/4)</span>
        </div>
        <div class="w-full bg-slate-100 rounded-full h-2 mb-1.5">
          <div class="bar-fill ${col.bar} h-2 rounded-full" style="width:0%" data-pct="${pct}"></div>
        </div>
        <p class="text-xs text-slate-500 leading-relaxed">${c.rationale}</p>
      </div>`;
  });

  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll('.bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  });

  // ── Tutor feedback (derived from scores)
  const sorted   = [...data.criteria].sort((a,b) => b.level - a.level);
  const top2     = sorted.slice(0, 2);
  const bottom2  = sorted.slice(-2);

  const strengths = top2.map(c =>
    `<strong>${c.criterion_name}</strong> (Level ${c.level}) — ${c.descriptor}`
  );
  const steps = bottom2.map((c, i) => {
    const next = Math.min(c.level + 1, 4);
    return `<li class="flex items-start gap-2">
      <span class="flex-shrink-0 w-5 h-5 rounded-full bg-amber-100 text-amber-800 font-bold text-xs flex items-center justify-center mt-0.5">${i+1}</span>
      <span>Improve <strong>${c.criterion_name}</strong>: aim for Level ${next} — "${c.descriptor}"</span>
    </li>`;
  });
  const questions = [
    `Does your essay clearly address all aspects of the prompt?`,
    `Where could you add one more piece of specific evidence or detail?`,
    `How could you strengthen the weakest part of your essay using the rubric descriptor as a guide?`
  ];

  document.getElementById('fb-strengths').innerHTML = strengths.map(s => `<li>${s}</li>`).join('');
  document.getElementById('fb-steps').innerHTML = steps.join('');
  document.getElementById('fb-questions').innerHTML = questions.map(q => `<li>${q}</li>`).join('');
}

// ── UI state helpers ─────────────────────────────────────────
function showState(state) {
  document.getElementById('res-empty').classList.toggle('hidden',    state !== 'empty');
  document.getElementById('res-loading').classList.toggle('hidden',  state !== 'loading');
  document.getElementById('res-results').classList.toggle('hidden',  state !== 'results');
  if (state === 'results') {
    document.getElementById('res-results').classList.add('flex');
  } else {
    document.getElementById('res-results').classList.remove('flex');
  }
  if (state === 'loading') {
    document.getElementById('res-loading').classList.add('flex');
  } else {
    document.getElementById('res-loading').classList.remove('flex');
  }
}

function showResultTab(tab) {
  ['review','feedback'].forEach(t => {
    const btn   = document.getElementById(`rtab-${t}`);
    const panel = document.getElementById(`rtab-panel-${t}`);
    if (t === tab) {
      btn.classList.add('tab-active'); btn.classList.remove('tab-inactive');
      panel.classList.remove('hidden');
    } else {
      btn.classList.remove('tab-active'); btn.classList.add('tab-inactive');
      panel.classList.add('hidden');
    }
  });
}

// ── Sidebar navigation ───────────────────────────────────────
function switchTab(tab) {
  ['dashboard','grade-level','analytics'].forEach(v => {
    document.getElementById(`view-${v}`).classList.toggle('hidden', v !== tab);
    const nav = document.getElementById(`nav-${v}`);
    nav.classList.toggle('sidebar-active', v === tab);
    nav.classList.toggle('hover:bg-slate-800/60', v !== tab);
  });
  document.getElementById('header-title').textContent =
    { dashboard: 'Dashboard', 'grade-level': 'Grade Level', analytics: 'Analytics' }[tab];
}

// ── Bootstrap ────────────────────────────────────────────────
selectGrade(6);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=_HTML)
