/* ==========================================================================
   The Architecture of Intelligence — Prof. Liang Li — Interactive Learning Hub
   Core Interactive Engine (Vanilla JavaScript)
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initMathRendering();
  initNavigation();
  initRegressionLab();
  initBayesLab();
  initInfoTheoryLab();
  initPerceptronXORLab();
  initActivationLab();
  initTrainingStepper();
  initAttentionLab();
  initScalingLab();
  initClusterHierarchyLab();
  initAgentLoopSimulator();
  initAgentCaseStudy();
  initGlossary();
  initReferencesArchive();
  initSpotlightSearch();
  initLightbox();
  initKinematicsLab();
});

/* ==========================================================================
   THEME TOGGLE — Light / Dark Mode
   ========================================================================== */
function initThemeToggle() {
  const btn = document.getElementById('themeToggleBtn');
  const label = document.getElementById('themeToggleLabel');
  const icon = btn ? btn.querySelector('.theme-toggle-icon') : null;

  // Restore saved preference
  const saved = localStorage.getItem('ai-edu-theme');
  if (saved === 'light') {
    document.body.classList.add('light-mode');
    if (icon) icon.textContent = '🌙';
    if (label) label.textContent = 'Dark';
  }

  if (!btn) return;

  btn.addEventListener('click', () => {
    const isLight = document.body.classList.toggle('light-mode');
    if (icon) icon.textContent = isLight ? '🌙' : '☀️';
    if (label) label.textContent = isLight ? 'Dark' : 'Light';
    localStorage.setItem('ai-edu-theme', isLight ? 'light' : 'dark');
    window.dispatchEvent(new CustomEvent('themeChanged', { detail: { isLight } }));
  });
}


/* ==========================================================================
   1. NAVIGATION & SCROLL TRACKING
   ========================================================================== */
function initNavigation() {
  const progressBar = document.getElementById('readingProgress');
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.course-part, .hero-section');

  window.addEventListener('scroll', () => {
    // Update progress bar
    const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    if (progressBar) {
      progressBar.style.width = scrolled + '%';
    }

    // Active nav link highlight
    let current = '';
    sections.forEach(sec => {
      const sectionTop = sec.offsetTop - 120;
      if (winScroll >= sectionTop) {
        current = sec.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

/* ==========================================================================
   2. PART 1: REGRESSION LAB (LINEAR & LOGISTIC)
   ========================================================================== */
function initRegressionLab() {
  const canvas = document.getElementById('regressionCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const slopeSlider = document.getElementById('regSlope');
  const biasSlider = document.getElementById('regBias');
  const slopeVal = document.getElementById('regSlopeVal');
  const biasVal = document.getElementById('regBiasVal');
  const mseVal = document.getElementById('regMSEVal');
  const autoStepBtn = document.getElementById('regAutoStepBtn');
  const resetBtn = document.getElementById('regResetBtn');
  const modeRadios = document.querySelectorAll('input[name="regMode"]');

  // Synthetic dataset: Study Hours (x in [1, 9]) vs Exam Score (y in [20, 95])
  const linearData = [
    { x: 1.5, y: 32 }, { x: 2.2, y: 38 }, { x: 3.0, y: 46 },
    { x: 3.8, y: 55 }, { x: 4.5, y: 58 }, { x: 5.2, y: 68 },
    { x: 6.0, y: 74 }, { x: 7.1, y: 82 }, { x: 7.8, y: 85 },
    { x: 8.5, y: 94 }
  ];

  // Binary classification: Hours vs Pass(1)/Fail(0)
  const logisticData = [
    { x: 1.2, y: 0 }, { x: 1.8, y: 0 }, { x: 2.5, y: 0 },
    { x: 3.2, y: 0 }, { x: 4.0, y: 0 }, { x: 4.6, y: 1 },
    { x: 5.5, y: 1 }, { x: 6.2, y: 1 }, { x: 7.0, y: 1 },
    { x: 8.2, y: 1 }
  ];

  let mode = 'linear';

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));
  }

  function computeMSE(w, b) {
    const data = mode === 'linear' ? linearData : logisticData;
    let sumErr = 0;
    data.forEach(pt => {
      const pred = mode === 'linear' ? (w * pt.x + b) : sigmoid(w * pt.x + b);
      sumErr += Math.pow(pred - pt.y, 2);
    });
    return (sumErr / data.length);
  }

  function render() {
    const w = parseFloat(slopeSlider.value);
    const b = parseFloat(biasSlider.value);
    slopeVal.textContent = w.toFixed(2);
    biasVal.textContent = b.toFixed(2);

    const mse = computeMSE(w, b);
    mseVal.textContent = mse.toFixed(2);

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Padding
    const pad = 45;
    const plotW = width - pad * 2;
    const plotH = height - pad * 2;

    // Coordinate transforms
    const xMin = 0, xMax = 10;
    const yMin = mode === 'linear' ? 0 : -0.1;
    const yMax = mode === 'linear' ? 105 : 1.1;

    function toCanvasX(x) {
      return pad + ((x - xMin) / (xMax - xMin)) * plotW;
    }
    function toCanvasY(y) {
      return height - pad - ((y - yMin) / (yMax - yMin)) * plotH;
    }

    // Draw Grid & Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= 10; x += 2) {
      const cx = toCanvasX(x);
      ctx.beginPath();
      ctx.moveTo(cx, pad);
      ctx.lineTo(cx, height - pad);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '11px Fira Code';
      ctx.fillText(x.toString(), cx - 4, height - pad + 18);
    }

    const ySteps = mode === 'linear' ? [0, 25, 50, 75, 100] : [0, 0.25, 0.5, 0.75, 1.0];
    ySteps.forEach(y => {
      const cy = toCanvasY(y);
      ctx.beginPath();
      ctx.moveTo(pad, cy);
      ctx.lineTo(width - pad, cy);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '11px Fira Code';
      ctx.fillText(y.toString(), pad - 32, cy + 4);
    });

    // Axis Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter';
    ctx.fillText('Study Hours (x)', width / 2 - 40, height - 10);

    ctx.save();
    ctx.translate(14, height / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(mode === 'linear' ? 'Score (y)' : 'P(Pass) (ŷ)', 0, 0);
    ctx.restore();

    // Draw Prediction Curve / Line
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    const resolution = 100;
    for (let i = 0; i <= resolution; i++) {
      const x = xMin + (i / resolution) * (xMax - xMin);
      const y = mode === 'linear' ? (w * x + b) : sigmoid(w * x + b);
      const cx = toCanvasX(x);
      const cy = toCanvasY(y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Draw Residuals & Data Points
    const data = mode === 'linear' ? linearData : logisticData;
    data.forEach(pt => {
      const cx = toCanvasX(pt.x);
      const cy = toCanvasY(pt.y);
      const predY = mode === 'linear' ? (w * pt.x + b) : sigmoid(w * pt.x + b);
      const cPredY = toCanvasY(predY);

      // Residual dashed line
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx, cPredY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Point
      ctx.fillStyle = '#a855f7';
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  slopeSlider.addEventListener('input', render);
  biasSlider.addEventListener('input', render);

  modeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      mode = e.target.value;
      if (mode === 'linear') {
        slopeSlider.min = "0";
        slopeSlider.max = "15";
        slopeSlider.step = "0.1";
        slopeSlider.value = "5";
        biasSlider.min = "-20";
        biasSlider.max = "60";
        biasSlider.step = "0.5";
        biasSlider.value = "10";
      } else {
        slopeSlider.min = "-2";
        slopeSlider.max = "5";
        slopeSlider.step = "0.1";
        slopeSlider.value = "0.5";
        biasSlider.min = "-10";
        biasSlider.max = "10";
        biasSlider.step = "0.1";
        biasSlider.value = "-1";
      }
      render();
    });
  });

  // Auto step gradient descent
  autoStepBtn.addEventListener('click', () => {
    const data = mode === 'linear' ? linearData : logisticData;
    let w = parseFloat(slopeSlider.value);
    let b = parseFloat(biasSlider.value);
    const alpha = mode === 'linear' ? 0.005 : 0.2;

    let gradW = 0;
    let gradB = 0;
    data.forEach(pt => {
      if (mode === 'linear') {
        const pred = w * pt.x + b;
        const err = pred - pt.y;
        gradW += (2 / data.length) * err * pt.x;
        gradB += (2 / data.length) * err;
      } else {
        const pred = sigmoid(w * pt.x + b);
        const err = pred - pt.y;
        gradW += err * pt.x;
        gradB += err;
      }
    });

    w -= alpha * gradW;
    b -= alpha * gradB;

    slopeSlider.value = w;
    biasSlider.value = b;
    render();
  });

  resetBtn.addEventListener('click', () => {
    if (mode === 'linear') {
      slopeSlider.value = "5";
      biasSlider.value = "10";
    } else {
      slopeSlider.value = "0.5";
      biasSlider.value = "-1";
    }
    render();
  });

  render();
}

/* ==========================================================================
   3. PART 1: BAYES' THEOREM & BASE-RATE FALLACY SIMULATOR (SECTION 4)
   ========================================================================== */
function initBayesLab() {
  const priorSlider = document.getElementById('bayesPrior');
  const sensSlider = document.getElementById('bayesSens');
  const fpSlider = document.getElementById('bayesFP');

  const priorVal = document.getElementById('bayesPriorVal');
  const sensVal = document.getElementById('bayesSensVal');
  const fpVal = document.getElementById('bayesFPVal');

  const postVal = document.getElementById('bayesPostVal');
  const totalPosVal = document.getElementById('bayesTotalPosVal');
  const tpCount = document.getElementById('bayesTPCount');
  const fpCount = document.getElementById('bayesFPCount');
  const fnCount = document.getElementById('bayesFNCount');
  const tnCount = document.getElementById('bayesTNCount');
  const insightText = document.getElementById('bayesInsightText');

  function update() {
    const pD = parseFloat(priorSlider.value); // Prior P(Cancer)
    const pSens = parseFloat(sensSlider.value); // Sensitivity P(+|Cancer)
    const pFP = parseFloat(fpSlider.value); // False Positive P(+|No Cancer)

    priorVal.textContent = (pD * 100).toFixed(1) + '%';
    sensVal.textContent = (pSens * 100).toFixed(1) + '%';
    fpVal.textContent = (pFP * 100).toFixed(1) + '%';

    const pNotD = 1 - pD;
    // Total Probability of positive test: P(+) = P(+|D)*P(D) + P(+|~D)*P(~D)
    const pPlus = (pSens * pD) + (pFP * pNotD);
    // Posterior: P(D|+) = [P(+|D) * P(D)] / P(+)
    const pPost = (pSens * pD) / pPlus;

    postVal.textContent = (pPost * 100).toFixed(1) + '%';
    totalPosVal.textContent = (pPlus * 100).toFixed(2) + '%';

    // 10,000 population breakdown
    const pop = 10000;
    const countD = Math.round(pop * pD);
    const countNotD = pop - countD;

    const tp = Math.round(countD * pSens);
    const fn = countD - tp;
    const fp = Math.round(countNotD * pFP);
    const tn = countNotD - fp;

    tpCount.textContent = tp.toLocaleString();
    fpCount.textContent = fp.toLocaleString();
    fnCount.textContent = fn.toLocaleString();
    tnCount.textContent = tn.toLocaleString();

    // Educational explanation of base-rate fallacy
    const ratio = (fp / Math.max(1, tp)).toFixed(1);
    if (pPost < 0.3) {
      insightText.innerHTML = `<strong>The Base-Rate Fallacy in Action:</strong> Even with <strong>${(pSens * 100).toFixed(0)}% sensitivity</strong>, because cancer is rare (prior = ${(pD * 100).toFixed(1)}%), there are <strong>${fp} false positives</strong> for every <strong>${tp} true positives</strong> (${ratio}:1 false alarm ratio). Thus, testing positive only implies a <strong>${(pPost * 100).toFixed(1)}%</strong> chance of actually having cancer!`;
    } else {
      insightText.innerHTML = `With higher disease prevalence (${(pD * 100).toFixed(1)}%), true positives (${tp}) outweigh false alarms (${fp}), boosting posterior confidence to <strong>${(pPost * 100).toFixed(1)}%</strong>.`;
    }
  }

  priorSlider.addEventListener('input', update);
  sensSlider.addEventListener('input', update);
  fpSlider.addEventListener('input', update);

  update();
}

/* ==========================================================================
   4. PART 1: INFORMATION THEORY EXPLORER (SECTION 5)
   ========================================================================== */
function initInfoTheoryLab() {
  const pSlider = document.getElementById('infoProbSlider');
  const pVal = document.getElementById('infoProbVal');
  const entropyVal = document.getElementById('infoEntropyVal');
  const canvas = document.getElementById('infoTheoryCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function update() {
    const p = parseFloat(pSlider.value);
    pVal.textContent = p.toFixed(2);

    // Binary Entropy H(X) = -p log2(p) - (1-p) log2(1-p)
    let H = 0;
    if (p > 0 && p < 1) {
      H = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
    }
    entropyVal.textContent = H.toFixed(3) + ' bits';

    // Draw curve
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = 40;
    const pw = w - pad * 2;
    const ph = h - pad * 2;

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, h - pad);
    ctx.stroke();

    // X-axis ticks
    [0, 0.25, 0.5, 0.75, 1.0].forEach(t => {
      const cx = pad + t * pw;
      ctx.fillStyle = '#64748b';
      ctx.font = '10px Fira Code';
      ctx.fillText(t.toFixed(2), cx - 10, h - pad + 16);
    });

    // Y-axis ticks
    [0, 0.5, 1.0].forEach(t => {
      const cy = h - pad - t * ph;
      ctx.fillStyle = '#64748b';
      ctx.font = '10px Fira Code';
      ctx.fillText(t.toFixed(1), pad - 26, cy + 4);
    });

    // Curve
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 1; i < 100; i++) {
      const prob = i / 100;
      const ent = -(prob * Math.log2(prob) + (1 - prob) * Math.log2(1 - prob));
      const cx = pad + prob * pw;
      const cy = h - pad - ent * ph;
      if (i === 1) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Active marker
    const markerX = pad + p * pw;
    const markerY = h - pad - H * ph;

    ctx.strokeStyle = '#a855f7';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(markerX, h - pad);
    ctx.lineTo(markerX, markerY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(markerX, markerY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  pSlider.addEventListener('input', update);
  update();
}

/* ==========================================================================
   5. PART 2: PERCEPTRON & THE XOR PROBLEM LAB (SECTION 7-8)
   ========================================================================== */
function initPerceptronXORLab() {
  const canvas = document.getElementById('xorCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const w1Slider = document.getElementById('xorW1');
  const w2Slider = document.getElementById('xorW2');
  const bSlider = document.getElementById('xorB');

  const w1Val = document.getElementById('xorW1Val');
  const w2Val = document.getElementById('xorW2Val');
  const bVal = document.getElementById('xorBVal');
  const xorAccuracy = document.getElementById('xorAccuracy');
  const mlpToggleBtn = document.getElementById('xorMLPToggle');
  const taskSelect = document.getElementById('xorTaskSelect');

  let useMLP = false;

  const tasks = {
    'AND': [
      { x1: 0, x2: 0, y: 0 },
      { x1: 0, x2: 1, y: 0 },
      { x1: 1, x2: 0, y: 0 },
      { x1: 1, x2: 1, y: 1 }
    ],
    'OR': [
      { x1: 0, x2: 0, y: 0 },
      { x1: 0, x2: 1, y: 1 },
      { x1: 1, x2: 0, y: 1 },
      { x1: 1, x2: 1, y: 1 }
    ],
    'XOR': [
      { x1: 0, x2: 0, y: 0 },
      { x1: 0, x2: 1, y: 1 },
      { x1: 1, x2: 0, y: 1 },
      { x1: 1, x2: 1, y: 0 }
    ]
  };

  function render() {
    const currentTask = taskSelect.value;
    const points = tasks[currentTask];
    const w1 = parseFloat(w1Slider.value);
    const w2 = parseFloat(w2Slider.value);
    const b = parseFloat(bSlider.value);

    w1Val.textContent = w1.toFixed(1);
    w2Val.textContent = w2.toFixed(1);
    bVal.textContent = b.toFixed(1);

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = 45;
    const pw = w - pad * 2;
    const ph = h - pad * 2;

    function toCX(x) { return pad + x * pw; }
    function toCY(y) { return h - pad - y * ph; }

    // Coordinate grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    [0, 0.5, 1.0].forEach(val => {
      const cx = toCX(val);
      const cy = toCY(val);
      ctx.beginPath();
      ctx.moveTo(cx, pad);
      ctx.lineTo(cx, h - pad);
      ctx.moveTo(pad, cy);
      ctx.lineTo(w - pad, cy);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '11px Fira Code';
      ctx.fillText(val.toString(), cx - 8, h - pad + 18);
      ctx.fillText(val.toString(), pad - 25, cy + 4);
    });

    let correctCount = 0;

    if (!useMLP) {
      // Single Linear Perceptron decision line: w1*x1 + w2*x2 + b = 0
      // => x2 = (-w1*x1 - b) / w2
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 3;
      ctx.beginPath();
      if (Math.abs(w2) > 0.001) {
        const yAt0 = (-b) / w2;
        const yAt1 = (-w1 - b) / w2;
        ctx.moveTo(toCX(0), toCY(yAt0));
        ctx.lineTo(toCX(1), toCY(yAt1));
      } else if (Math.abs(w1) > 0.001) {
        const xAt0 = -b / w1;
        ctx.moveTo(toCX(xAt0), toCY(0));
        ctx.lineTo(toCX(xAt0), toCY(1));
      }
      ctx.stroke();

      // Check predictions: 1 if w1*x1 + w2*x2 + b >= 0 else 0
      points.forEach(pt => {
        const score = w1 * pt.x1 + w2 * pt.x2 + b;
        const pred = score >= 0 ? 1 : 0;
        if (pred === pt.y) correctCount++;
      });
    } else {
      // MLP with 2 hidden units solving XOR
      // Line 1: x1 + x2 - 0.5 = 0 (OR gate)
      // Line 2: x1 + x2 - 1.5 = 0 (NAND gate)
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(toCX(0), toCY(0.5));
      ctx.lineTo(toCX(0.5), toCY(0));
      ctx.stroke();

      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(toCX(0.5), toCY(1));
      ctx.lineTo(toCX(1), toCY(0.5));
      ctx.stroke();

      // In 2-layer MLP mode, it correctly separates all 4 points
      correctCount = 4;
    }

    // Draw Points
    points.forEach(pt => {
      const cx = toCX(pt.x1);
      const cy = toCY(pt.x2);

      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.fillStyle = pt.y === 1 ? '#10b981' : '#f43f5e';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px Fira Code';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`y=${pt.y}`, cx, cy - 18);
    });

    // Update Accuracy
    const accPercent = (correctCount / 4) * 100;
    xorAccuracy.textContent = `${accPercent.toFixed(0)}% (${correctCount}/4)`;
    if (currentTask === 'XOR' && !useMLP) {
      xorAccuracy.style.color = '#f43f5e';
    } else {
      xorAccuracy.style.color = '#10b981';
    }
  }

  w1Slider.addEventListener('input', render);
  w2Slider.addEventListener('input', render);
  bSlider.addEventListener('input', render);
  taskSelect.addEventListener('change', () => {
    useMLP = false;
    mlpToggleBtn.textContent = 'Add Hidden Layer (Solve XOR)';
    if (taskSelect.value === 'AND') {
      w1Slider.value = '2.0';
      w2Slider.value = '2.0';
      bSlider.value = '-3.0';
    } else if (taskSelect.value === 'OR') {
      w1Slider.value = '2.0';
      w2Slider.value = '2.0';
      bSlider.value = '-1.0';
    } else {
      w1Slider.value = '1.0';
      w2Slider.value = '1.0';
      bSlider.value = '-1.0';
    }
    render();
  });

  mlpToggleBtn.addEventListener('click', () => {
    useMLP = !useMLP;
    mlpToggleBtn.textContent = useMLP ? 'Revert to Single Linear Neuron' : 'Add Hidden Layer (Solve XOR)';
    render();
  });

  render();
}

/* ==========================================================================
   6. PART 2: ACTIVATION FUNCTION LAB (SECTION 8)
   ========================================================================== */
function initActivationLab() {
  const canvas = document.getElementById('activationCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const actSelect = document.getElementById('actFuncSelect');
  const actFormula = document.getElementById('actFormulaText');
  const actNote = document.getElementById('actNoteText');

  const funcs = {
    'relu': {
      f: z => Math.max(0, z),
      df: z => (z > 0 ? 1 : 0),
      formula: 'f(z) = \\max(0, z), \\quad f\'(z) = \\begin{cases} 1 & z > 0 \\\\ 0 & z \\le 0 \\end{cases}',
      note: 'Most popular in deep feed-forward & CNNs. Solves vanishing gradients for positive inputs, but subject to "Dying ReLU" for persistent negative values.'
    },
    'gelu': {
      f: z => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * Math.pow(z, 3)))),
      df: z => 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * Math.pow(z, 3)))) + 0.5 * z * (1 - Math.pow(Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * Math.pow(z, 3))), 2)) * Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * Math.pow(z, 2)),
      formula: 'f(z) = z \\cdot \\Phi(z) \\approx 0.5z \\left(1 + \\tanh\\left(\\sqrt{2/\\pi}(z + 0.044715z^3)\\right)\\right)',
      note: 'The modern standard in Transformer architectures (BERT, GPT, Claude). Smoothly non-linear everywhere, introducing probabilistic dropout behavior into the activation.'
    },
    'sigmoid': {
      f: z => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z)))),
      df: z => {
        const s = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));
        return s * (1 - s);
      },
      formula: 'f(z) = \\frac{1}{1 + e^{-z}}, \\quad f\'(z) = f(z)(1 - f(z)) \\le 0.25',
      note: 'Compresses output to (0, 1). Essential for binary output probabilities, but causes severe Vanishing Gradients in deep hidden layers since max derivative is only 0.25.'
    },
    'tanh': {
      f: z => Math.tanh(z),
      df: z => 1 - Math.pow(Math.tanh(z), 2),
      formula: 'f(z) = \\tanh(z), \\quad f\'(z) = 1 - f(z)^2',
      note: 'Zero-centered range (-1, 1). Preferred over Sigmoid in RNN internal hidden states, though still saturates at extremes.'
    }
  };

  function render() {
    const key = actSelect.value;
    const cur = funcs[key];
    if (typeof katex !== 'undefined' && actFormula) {
      try {
        katex.render(cur.formula, actFormula, { throwOnError: false, displayMode: true });
      } catch (e) {
        actFormula.textContent = cur.formula;
      }
    } else if (actFormula) {
      actFormula.textContent = cur.formula;
    }
    actNote.textContent = cur.note;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const pad = 40;
    const pw = w - pad * 2;
    const ph = h - pad * 2;

    const zMin = -5, zMax = 5;
    const yMin = -1.2, yMax = 4.0;

    function toCX(z) { return pad + ((z - zMin) / (zMax - zMin)) * pw; }
    function toCY(y) { return h - pad - ((y - yMin) / (yMax - yMin)) * ph; }

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toCX(0), pad);
    ctx.lineTo(toCX(0), h - pad);
    ctx.moveTo(pad, toCY(0));
    ctx.lineTo(w - pad, toCY(0));
    ctx.stroke();

    // Activation Curve f(z)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const z = zMin + (i / 200) * (zMax - zMin);
      const y = cur.f(z);
      const cx = toCX(z);
      const cy = toCY(y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Derivative Curve f'(z)
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const z = zMin + (i / 200) * (zMax - zMin);
      const y = cur.df(z);
      const cx = toCX(z);
      const cy = toCY(y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    ctx.fillStyle = '#38bdf8';
    ctx.font = '11px Fira Code';
    ctx.fillText('— f(z) Activation', pad + 10, pad + 15);
    ctx.fillStyle = '#a855f7';
    ctx.fillText('-- f\'(z) Derivative', pad + 10, pad + 32);
  }

  actSelect.addEventListener('change', render);
  render();
}

/* ==========================================================================
   7. PART 2: 5-STEP TRAINING LOOP STEPPER (SECTION 9)
   ========================================================================== */
function initTrainingStepper() {
  const stepBtns = document.querySelectorAll('.training-step-btn');
  const stepTitle = document.getElementById('stepCardTitle');
  const stepDesc = document.getElementById('stepCardDesc');
  const stepMath = document.getElementById('stepCardMath');
  const stepCode = document.getElementById('stepCardCode');

  const stepsData = [
    {
      title: 'Step 1 — Forward Propagation (Prediction Phase)',
      desc: 'Input data flows through the layers. Each layer calculates an affine transformation z^[l] = W^[l] a^[l-1] + b^[l], followed by an activation function a^[l] = σ(z^[l]). The final output represents the model\'s prediction ŷ.',
      math: 'z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \\quad a^{[l]} = \\sigma(z^{[l]})',
      code: 'def forward_propagation(X, parameters):\n    # Layer 1\n    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]\n    A1 = relu(Z1)\n    # Layer 2 (Output)\n    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]\n    A2 = sigmoid(Z2)\n    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}'
    },
    {
      title: 'Step 2 — Loss Calculation (Error Measurement)',
      desc: 'Compares the model\'s prediction ŷ against the ground truth y using an objective function. For classification, Cross-Entropy Loss measures the information-theoretic penalty.',
      math: 'L_{CE}(y, \\hat{y}) = -\\left[ y \\log(\\hat{y}) + (1 - y) \\log(1 - \\hat{y}) \\right]',
      code: 'def compute_cross_entropy_loss(A2, Y):\n    m = Y.shape[1]\n    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)\n    cost = -np.sum(logprobs) / m\n    return np.squeeze(cost)'
    },
    {
      title: 'Step 3 — Backpropagation (Error Correction via Chain Rule)',
      desc: 'Applies the multivariate chain rule from calculus to propagate the error backward from output to input, computing the exact gradient of the loss with respect to every weight and bias: ∂L/∂W and ∂L/∂b.',
      math: '\\frac{\\partial L}{\\partial W^{[l]}} = \\frac{\\partial L}{\\partial a^{[l]}} \\cdot \\frac{\\partial a^{[l]}}{\\partial z^{[l]}} \\cdot \\frac{\\partial z^{[l]}}{\\partial W^{[l]}} = \\delta^{[l]} (a^{[l-1]})^T',
      code: 'def backpropagation(parameters, cache, X, Y):\n    m = X.shape[1]\n    dZ2 = cache["A2"] - Y\n    dW2 = (1 / m) * np.dot(dZ2, cache["A1"].T)\n    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)\n    \n    dZ1 = np.dot(parameters["W2"].T, dZ2) * relu_derivative(cache["Z1"])\n    dW1 = (1 / m) * np.dot(dZ1, X.T)\n    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)\n    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}'
    },
    {
      title: 'Step 4 — Gradient Descent (Weight Optimization)',
      desc: 'Updates each parameter by taking a step in the opposite direction of the gradient, scaled by learning rate α. Advanced optimizers like Adam incorporate momentum and adaptive second moments.',
      math: 'W^{[l]} \\leftarrow W^{[l]} - \\alpha \\frac{\\partial L}{\\partial W^{[l]}}, \\quad b^{[l]} \\leftarrow b^{[l]} - \\alpha \\frac{\\partial L}{\\partial b^{[l]}}',
      code: 'def update_parameters(parameters, grads, learning_rate=0.01):\n    parameters["W1"] -= learning_rate * grads["dW1"]\n    parameters["b1"] -= learning_rate * grads["db1"]\n    parameters["W2"] -= learning_rate * grads["dW2"]\n    parameters["b2"] -= learning_rate * grads["db2"]\n    return parameters'
    },
    {
      title: 'Step 5 — Iteration (Epochs & Convergence)',
      desc: 'The complete sequence of forward prop, loss calculation, backprop, and parameter updates repeats across many passes (epochs) through the dataset until the loss converges to a minimum.',
      math: '\\min_{\\theta} \\frac{1}{N} \\sum_{i=1}^N L(y_i, f(x_i; \\theta))',
      code: 'for epoch in range(num_epochs):\n    # 1. Forward pass\n    A2, cache = forward_propagation(X, parameters)\n    # 2. Compute cost\n    cost = compute_cross_entropy_loss(A2, Y)\n    # 3. Backward pass\n    grads = backpropagation(parameters, cache, X, Y)\n    # 4. Update weights\n    parameters = update_parameters(parameters, grads, alpha)\n    if epoch % 100 == 0: print(f"Epoch {epoch}: Loss = {cost:.4f}")'
    }
  ];

  function showStep(idx) {
    stepBtns.forEach((b, i) => {
      b.classList.toggle('active', i === idx);
    });
    const d = stepsData[idx];
    stepTitle.textContent = d.title;
    stepDesc.textContent = d.desc;
    if (typeof katex !== 'undefined' && stepMath) {
      try {
        katex.render(d.math, stepMath, { throwOnError: false, displayMode: true });
      } catch (e) {
        stepMath.textContent = d.math;
      }
    } else if (stepMath) {
      stepMath.textContent = d.math;
    }
    stepCode.textContent = d.code;
  }

  stepBtns.forEach((btn, idx) => {
    btn.addEventListener('click', () => showStep(idx));
  });

  showStep(0);
}

/* ==========================================================================
   8. PART 3: ATTENTION REVOLUTION LAB (SECTION 12)
   ========================================================================== */
function initAttentionLab() {
  const sentence = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"];
  const tokenListEl = document.getElementById('attnTokenList');
  const tokenDetailsEl = document.getElementById('attnTokenDetails');
  const heatmapGridEl = document.getElementById('attnHeatmapGrid');
  if (!tokenListEl || !heatmapGridEl) return;

  // Synthetic attention weights matrix (11x11)
  // "it" attends heavily to "animal" (index 1) and "tired" (index 10)
  const attnMatrix = [
    // The
    [0.45, 0.35, 0.05, 0.02, 0.02, 0.05, 0.01, 0.02, 0.01, 0.01, 0.01],
    // animal
    [0.10, 0.55, 0.05, 0.08, 0.02, 0.05, 0.02, 0.03, 0.02, 0.02, 0.06],
    // didn't
    [0.02, 0.08, 0.40, 0.30, 0.02, 0.05, 0.04, 0.03, 0.02, 0.01, 0.03],
    // cross
    [0.02, 0.15, 0.10, 0.45, 0.05, 0.15, 0.02, 0.02, 0.01, 0.01, 0.02],
    // the
    [0.02, 0.03, 0.02, 0.05, 0.35, 0.45, 0.02, 0.02, 0.01, 0.01, 0.02],
    // street
    [0.02, 0.05, 0.02, 0.20, 0.15, 0.48, 0.02, 0.02, 0.01, 0.01, 0.02],
    // because
    [0.02, 0.04, 0.05, 0.05, 0.02, 0.04, 0.45, 0.15, 0.08, 0.05, 0.05],
    // it (KEY FOCUS)
    [0.01, 0.58, 0.02, 0.04, 0.01, 0.08, 0.04, 0.05, 0.03, 0.02, 0.12],
    // was
    [0.01, 0.05, 0.02, 0.02, 0.01, 0.02, 0.05, 0.40, 0.30, 0.05, 0.07],
    // too
    [0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.02, 0.05, 0.05, 0.35, 0.44],
    // tired
    [0.01, 0.38, 0.02, 0.02, 0.01, 0.02, 0.04, 0.22, 0.05, 0.15, 0.08]
  ];

  let currentSelectedIdx = 7;

  function renderHeatmap(selectedIdx) {
    if (selectedIdx !== undefined) currentSelectedIdx = selectedIdx;
    const isLight = document.body.classList.contains('light-mode');
    heatmapGridEl.innerHTML = '';
    heatmapGridEl.style.gridTemplateColumns = `repeat(${sentence.length}, 1fr)`;

    for (let r = 0; r < sentence.length; r++) {
      for (let c = 0; c < sentence.length; c++) {
        const val = attnMatrix[r][c];
        const cell = document.createElement('div');
        cell.className = 'attention-cell';
        const isSelectedRow = (r === currentSelectedIdx);
        const percent = Math.round(val * 100);

        if (isLight) {
          // LIGHT MODE PALETTE - Crisp, high contrast, readable numbers
          if (isSelectedRow) {
            if (val >= 0.40) {
              cell.style.backgroundColor = '#0284c7';
              cell.style.color = '#ffffff';
              cell.style.border = '2px solid #0369a1';
              cell.style.fontWeight = '700';
            } else if (val >= 0.20) {
              cell.style.backgroundColor = '#38bdf8';
              cell.style.color = '#082f49';
              cell.style.border = '1px solid #0284c7';
              cell.style.fontWeight = '700';
            } else if (val >= 0.08) {
              cell.style.backgroundColor = '#bae6fd';
              cell.style.color = '#0369a1';
              cell.style.border = '1px solid #7dd3fc';
              cell.style.fontWeight = '600';
            } else {
              cell.style.backgroundColor = '#f0f9ff';
              cell.style.color = '#0284c7';
              cell.style.border = '1px solid #e0f2fe';
            }
          } else {
            // Unselected row: Crisp Purple progression
            if (val >= 0.40) {
              cell.style.backgroundColor = '#7c3aed';
              cell.style.color = '#ffffff';
              cell.style.border = '2px solid #6d28d9';
              cell.style.fontWeight = '700';
            } else if (val >= 0.20) {
              cell.style.backgroundColor = '#c084fc';
              cell.style.color = '#3b0764';
              cell.style.border = '1px solid #a855f7';
              cell.style.fontWeight = '700';
            } else if (val >= 0.08) {
              cell.style.backgroundColor = '#e9d5ff';
              cell.style.color = '#581c87';
              cell.style.border = '1px solid #d8b4fe';
              cell.style.fontWeight = '600';
            } else {
              cell.style.backgroundColor = '#ffffff';
              cell.style.color = '#64748b';
              cell.style.border = '1px solid #e2e8f0';
            }
          }
        } else {
          // DARK MODE PALETTE - Glowing neon cyber aesthetics
          if (isSelectedRow) {
            if (val >= 0.40) {
              cell.style.backgroundColor = '#0284c7';
              cell.style.color = '#ffffff';
              cell.style.border = '2px solid #38bdf8';
              cell.style.fontWeight = '700';
            } else if (val >= 0.20) {
              cell.style.backgroundColor = 'rgba(56, 189, 248, 0.75)';
              cell.style.color = '#ffffff';
              cell.style.border = '1px solid #38bdf8';
              cell.style.fontWeight = '600';
            } else if (val >= 0.08) {
              cell.style.backgroundColor = 'rgba(56, 189, 248, 0.35)';
              cell.style.color = '#e0f2fe';
              cell.style.border = '1px solid rgba(56, 189, 248, 0.5)';
            } else {
              cell.style.backgroundColor = 'rgba(14, 165, 233, 0.12)';
              cell.style.color = '#7dd3fc';
              cell.style.border = '1px solid rgba(56, 189, 248, 0.2)';
            }
          } else {
            if (val >= 0.40) {
              cell.style.backgroundColor = '#9333ea';
              cell.style.color = '#ffffff';
              cell.style.border = '2px solid #c084fc';
              cell.style.fontWeight = '700';
            } else if (val >= 0.20) {
              cell.style.backgroundColor = 'rgba(168, 85, 247, 0.7)';
              cell.style.color = '#ffffff';
              cell.style.border = '1px solid #a855f7';
              cell.style.fontWeight = '600';
            } else if (val >= 0.08) {
              cell.style.backgroundColor = 'rgba(168, 85, 247, 0.35)';
              cell.style.color = '#f3e8ff';
              cell.style.border = '1px solid rgba(168, 85, 247, 0.5)';
            } else {
              cell.style.backgroundColor = 'rgba(30, 41, 59, 0.6)';
              cell.style.color = '#64748b';
              cell.style.border = '1px solid rgba(255, 255, 255, 0.05)';
            }
          }
        }

        cell.title = `Query "${sentence[r]}" (row ${r}) attends to Key "${sentence[c]}" (col ${c}): ${percent}%`;
        cell.textContent = percent;

        cell.addEventListener('click', () => {
          selectToken(r);
        });

        heatmapGridEl.appendChild(cell);
      }
    }
  }

  function selectToken(idx) {
    const word = sentence[idx];
    currentSelectedIdx = idx;
    // Update token list buttons
    document.querySelectorAll('.attn-token-pill').forEach((btn, i) => {
      btn.classList.toggle('active', i === idx);
    });

    renderHeatmap(idx);

    // Rank top attended tokens for this query
    const row = attnMatrix[idx];
    const ranked = sentence
      .map((tok, i) => ({ tok, weight: row[i] }))
      .sort((a, b) => b.weight - a.weight);

    let detailsHtml = `<strong>Query Token: "${word}"</strong> (Position ${idx})<br>`;
    detailsHtml += `<span style="font-size:0.85rem; color:var(--text-muted);">Highest attention weights computed via \\(\\text{Softmax}(q_{${idx}} K^T / \\sqrt{d_k})\\):</span><ul style="margin-top:0.5rem; padding-left:1.2rem; font-family:var(--font-mono); font-size:0.85rem;">`;
    ranked.slice(0, 3).forEach(item => {
      detailsHtml += `<li><strong>"${item.tok}"</strong>: ${(item.weight * 100).toFixed(1)}%</li>`;
    });
    detailsHtml += `</ul>`;

    if (word === "it") {
      detailsHtml += `<div style="margin-top:0.5rem; padding:0.5rem; background:rgba(56,189,248,0.12); border-left:3px solid #38bdf8; font-size:0.82rem; border-radius:4px;">
        💡 <strong>Key Linguistic Discovery:</strong> The model resolves coreference disambiguation! The ambiguous pronoun <em>"it"</em> assigns <strong>58% attention</strong> to <em>"animal"</em> because <em>"tired"</em> is an attribute of animate beings, not the <em>"street"</em>.
      </div>`;
    }

    tokenDetailsEl.innerHTML = detailsHtml;
  }

  // Populate pills
  tokenListEl.innerHTML = '';
  sentence.forEach((tok, i) => {
    const pill = document.createElement('button');
    pill.className = 'btn btn-secondary btn-sm attn-token-pill';
    pill.style.fontFamily = 'var(--font-mono)';
    pill.textContent = tok;
    pill.addEventListener('click', () => selectToken(i));
    tokenListEl.appendChild(pill);
  });

  // Re-render heatmap on theme toggle
  window.addEventListener('themeChanged', () => {
    renderHeatmap(currentSelectedIdx);
  });

  // Default select "it" (index 7)
  selectToken(7);
}

/* ==========================================================================
   9. PART 4: SCALING LAWS & CHINCHILLA FRONTIER LAB (SECTION 13)
   ========================================================================== */
function initScalingLab() {
  const paramSlider = document.getElementById('scalingParams');
  const paramVal = document.getElementById('scalingParamsVal');
  const tokensVal = document.getElementById('scalingTokensVal');
  const flopsVal = document.getElementById('scalingFlopsVal');
  const storageVal = document.getElementById('scalingStorageVal');

  function update() {
    const paramsB = parseFloat(paramSlider.value); // In Billions
    paramVal.textContent = paramsB >= 1000 ? `${(paramsB / 1000).toFixed(1)}T (${paramsB}B)` : `${paramsB}B`;

    // Chinchilla compute-optimal rule: Tokens D = 20 * N
    const optimalTokensB = paramsB * 20;
    tokensVal.textContent = optimalTokensB >= 1000 ? `${(optimalTokensB / 1000).toFixed(2)} Trillion` : `${optimalTokensB} Billion`;

    // Compute FLOPs C ≈ 6 * N * D
    // N in units, D in units
    // N * D in 10^18
    const flops10_24 = (6 * (paramsB * 1e9) * (optimalTokensB * 1e9)) / 1e24;
    flopsVal.textContent = `${flops10_24.toFixed(2)} × 10²⁴ FLOPs`;

    // Uncompressed FP16/BF16 weights storage: 2 bytes per param
    const gigabytes = paramsB * 2;
    if (gigabytes >= 1000) {
      storageVal.textContent = `${(gigabytes / 1000).toFixed(2)} TB (weights only)`;
    } else {
      storageVal.textContent = `${gigabytes.toFixed(0)} GB`;
    }
  }

  paramSlider.addEventListener('input', update);
  update();
}

/* ==========================================================================
   10. PART 5: COMPUTE SCALE HIERARCHY & BANDWIDTH LAB (SECTION 15-16)
   ========================================================================== */
function initClusterHierarchyLab() {
  const pills = document.querySelectorAll('.scale-stage-pill');
  const nameEl = document.getElementById('scaleLevelName');
  const descEl = document.getElementById('scaleLevelDesc');
  const specsEl = document.getElementById('scaleLevelSpecs');
  const bwFill = document.getElementById('scaleBwFill');
  const bwVal = document.getElementById('scaleBwVal');

  const hierarchyData = [
    {
      name: 'Level 1: Single NVIDIA H200 SXM5 GPU',
      desc: 'The atomic engine of modern deep learning. Contains 80 billion transistors built on TSMC 4N process, 132 Streaming Multiprocessors (SMs), 4th-Gen Tensor Cores with FP8 Transformer Engine, and 141GB ultra-fast HBM3e memory.',
      specs: '• Internal Memory Bandwidth: 4,800 GB/s (4.8 TB/s)\n• Memory Capacity: 141GB HBM3e (1.76x of H100)\n• Power Envelope (TDP): 700 Watts\n• Tensor Throughput: ~2,000 TFLOPs (FP8)\n• Interconnect: 900 GB/s NVLink 4.0 ports',
      bandwidth: 4800,
      maxBw: 4800,
      displayBw: '4,800 GB/s (HBM3e)'
    },
    {
      name: 'Level 2: 8-GPU Server (HGX H200 Baseboard)',
      desc: 'Aggregates 8 H200 GPUs into an all-to-all NVLink mesh via 4 NVSwitches on the baseboard. Provides 1.1TB aggregate HBM3e memory at wire speed.',
      specs: '• Inter-GPU Bandwidth: 900 GB/s (Bidirectional NVLink 4)\n• Aggregate NVLink Throughput: 7.2 TB/s\n• Total Memory: 1,128 GB HBM3e\n• Host Connectivity: Dual Intel Xeon/AMD EPYC CPUs + PCIe Gen 5\n• Power: ~10.2 Kilowatts per server',
      bandwidth: 900,
      maxBw: 4800,
      displayBw: '900 GB/s (NVLink 4)'
    },
    {
      name: 'Level 3: GPU Rack (2 Servers = 16 GPUs)',
      desc: 'Combines two 8-GPU servers with dedicated top-of-rack InfiniBand switches, power distribution units (PDUs), and liquid-cooling manifolds. Standard 42U rack format.',
      specs: '• Total GPUs: 16 H200 GPUs\n• Total Memory: 2,256 GB HBM3e\n• Network: 8x 400 Gb/s Quantum-2 InfiniBand CX7 NICs per server\n• Power: ~20.4 Kilowatts',
      bandwidth: 400,
      maxBw: 4800,
      displayBw: '400 GB/s (InfiniBand intra-rack)'
    },
    {
      name: 'Level 4: Full GPU Pod (192 Racks = 3,072 GPUs)',
      desc: 'The architectural SuperPOD boundary. 192 racks organized into a non-blocking fat-tree InfiniBand network fabric. As signals travel across optical transceivers and switches, inter-GPU bandwidth drops to roughly 50 GB/s.',
      specs: '• Total GPUs: 3,072 H200s\n• Pod Interconnect: ~50 GB/s per GPU across boundary\n• Aggregate Pod Memory: 433.1 Terabytes HBM3e\n• Power: ~4 Megawatts',
      bandwidth: 50,
      maxBw: 4800,
      displayBw: '50 GB/s (Fat-Tree Fabric)'
    },
    {
      name: 'Level 5: Hyperscale GPU Cluster (8 Pods = 24,576 GPUs)',
      desc: 'The complete frontier-training cluster acting as "one giant compute engine" capable of training 18-trillion token foundation models at 24+ ExaFLOPs throughput with massive 3.46 Petabyte HBM3e pool.',
      specs: '• Total GPUs: 24,576 H200s\n• FP32 Cores: 415 Million\n• Tensor Cores: 13 Million\n• Combined Throughput: 24+ ExaFLOPs (FP8)\n• Combined GPU Memory: 3.46 Petabytes HBM3e\n• Cross-Cluster Bandwidth: ~50 GB/s (~96x collapse from on-chip HBM3e)',
      bandwidth: 50,
      maxBw: 4800,
      displayBw: '~50 GB/s (Cluster Outer)'
    }
  ];

  function showLevel(idx) {
    pills.forEach((p, i) => p.classList.toggle('active', i === idx));
    const d = hierarchyData[idx];
    nameEl.textContent = d.name;
    descEl.textContent = d.desc;
    specsEl.textContent = d.specs;

    // Update bandwidth bar
    const pct = (d.bandwidth / d.maxBw) * 100;
    bwFill.style.width = `${pct}%`;
    bwFill.style.backgroundColor = idx === 0 ? '#38bdf8' : (idx === 1 ? '#a855f7' : (idx <= 3 ? '#f59e0b' : '#f43f5e'));
    bwVal.textContent = d.displayBw;
  }

  pills.forEach((pill, i) => {
    pill.addEventListener('click', () => showLevel(i));
  });

  showLevel(0);
}

/* ==========================================================================
   11. PART 6: AGENTIC AI LOOP SIMULATOR (SECTION 19)
   ========================================================================== */
function initAgentLoopSimulator() {
  const prevBtn = document.getElementById('agentPrevBtn');
  const nextBtn = document.getElementById('agentNextBtn');
  const playBtn = document.getElementById('agentPlayBtn');
  const resetBtn = document.getElementById('agentResetBtn');
  const terminalContent = document.getElementById('agentTerminalOutput');
  const nodes = document.querySelectorAll('.agent-node');

  let currentStep = 0;
  let isPlaying = false;
  let playInterval = null;

  const loopSteps = [
    {
      phase: 0, // PLAN
      badge: 'Step 1: Goal Intake & Planning',
      terminal: [
        '<span class="term-prompt">user@system:~$</span> run_agent --goal "Implement in-place Quicksort in Python and verify all edge cases"',
        '<span class="term-comment">[AGENT BRAIN / LLM] Decomposing high-level objective:</span>',
        '  1. Write recursive quicksort function using in-place partitioning.',
        '  2. Choose pivot strategy: Lomuto partitioning.',
        '  3. Write comprehensive unit test suite: empty array, single element, duplicates, already sorted.',
        '  4. Execute tests inside sandboxed environment and verify correctness.'
      ]
    },
    {
      phase: 1, // EXECUTE
      badge: 'Step 2: Execution & Tool Calling',
      terminal: [
        '<span class="term-tool">[TOOL CALL] code_sandbox.run_python(code=...)</span>',
        '<pre style="color:#e2e8f0; margin:0.4rem 0;">def quicksort(arr, low, high):\n    if low < high:\n        pi = partition(arr, low, high)\n        quicksort(arr, low, pi - 1)\n        quicksort(arr, pi + 1, high)\n\ndef partition(arr, low, high):\n    pivot = arr[high]\n    i = low - 1\n    for j in range(low, high):\n        if arr[j] <= pivot: # Note duplicate handling\n            i += 1\n            arr[i], arr[j] = arr[j], arr[i]\n    arr[i + 1], arr[high] = arr[high], arr[i + 1]\n    return i + 1</pre>',
        '<span class="term-comment">[SANDBOX] Invoking test suite with test array: [3, 1, 3, 2, 3]...</span>'
      ]
    },
    {
      phase: 2, // EVALUATE
      badge: 'Step 3: Evaluation & Error Detection',
      terminal: [
        '<span class="term-tool">[SANDBOX STDOUT]</span> Running test cases...',
        '  Test 1 (Standard): [4, 2, 8, 1] -> [1, 2, 4, 8]  [PASS]',
        '  Test 2 (Empty): [] -> []                         [PASS]',
        '<span class="term-error">  Test 3 (Duplicates): [3, 1, 3, 2, 3] -> FAILED</span>',
        '<span class="term-error">[ERROR TRACEBACK] RecursionError: maximum recursion depth exceeded in comparison</span>',
        '<span class="term-comment">[EVALUATOR] Goal NOT met: 1 of 3 tests failed. Initiating reflection loop.</span>'
      ]
    },
    {
      phase: 3, // REFLECT
      badge: 'Step 4: Reflection & Self-Correction',
      terminal: [
        '<span class="term-comment">[AGENT REFLECTION ENGINE] Analyzing error state:</span>',
        '  • <strong>Root Cause Diagnosis:</strong> When all elements are identical or duplicates match the pivot, the Lomuto partition places pivot at high, resulting in an unbalanced partition with no reduction in subproblem size (pi = high).',
        '  • <strong>Self-Correction Strategy:</strong> Switch to 3-Way QuickSort (Dutch National Flag partitioning) to isolate elements equal to pivot in O(N) time.',
        '  • <strong>Replanning:</strong> Rewrite partition logic to return `[lt, gt]` range and recurse only on `low .. lt-1` and `gt+1 .. high`.'
      ]
    },
    {
      phase: 1, // RE-EXECUTE
      badge: 'Step 5: Re-Execution with Refined Architecture',
      terminal: [
        '<span class="term-tool">[TOOL CALL] code_sandbox.run_python(code=quicksort_3way)</span>',
        '<span class="term-comment">[SANDBOX] Re-running full test suite including edge cases:</span>',
        '  Test 1 (Standard): [4, 2, 8, 1]          -> [1, 2, 4, 8]  [PASS]',
        '  Test 2 (Empty Array): []                 -> []            [PASS]',
        '  Test 3 (All Duplicates): [3, 3, 3, 3]     -> [3, 3, 3, 3]  [PASS]',
        '  Test 4 (Repeated Keys): [3, 1, 3, 2, 3]  -> [1, 2, 3, 3, 3] [PASS]',
        '  Test 5 (Reverse Sorted): [9, 8, 7]       -> [7, 8, 9]     [PASS]'
      ]
    },
    {
      phase: 0, // GOAL ACHIEVED
      badge: 'Step 6: Goal Accomplished & Verified Deliverable',
      terminal: [
        '<span class="term-success">[STATUS: SUCCESS] All 5 unit test cases passed with zero errors.</span>',
        '<span class="term-prompt">agent@system:~$</span> task_complete --status 200 --latency 1.4s',
        '<div style="margin-top:0.5rem; padding:0.6rem; background:rgba(16,185,129,0.15); border:1px solid #10b981; border-radius:6px; color:#d1fae5;">' +
        '✓ <strong>Autonomous Loop Complete:</strong> The system successfully moved from high-level goal → execution → failure detection → mathematical reflection → corrected deliverable without human intervention.' +
        '</div>'
      ]
    }
  ];

  function renderStep(idx) {
    const step = loopSteps[idx];
    // Update node highlights
    nodes.forEach((n, i) => {
      n.classList.toggle('active', i === step.phase);
    });

    // Terminal log
    terminalContent.innerHTML = `
      <div style="font-family:var(--font-mono); font-size:0.75rem; color:#38bdf8; margin-bottom:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">
        ${step.badge} (${idx + 1} / ${loopSteps.length})
      </div>
      ${step.terminal.map(line => `<div class="term-line">${line}</div>`).join('')}
    `;
    terminalContent.scrollTop = terminalContent.scrollHeight;

    prevBtn.disabled = (idx === 0);
    nextBtn.disabled = (idx === loopSteps.length - 1);
  }

  nextBtn.addEventListener('click', () => {
    if (currentStep < loopSteps.length - 1) {
      currentStep++;
      renderStep(currentStep);
    }
  });

  prevBtn.addEventListener('click', () => {
    if (currentStep > 0) {
      currentStep--;
      renderStep(currentStep);
    }
  });

  resetBtn.addEventListener('click', () => {
    stopPlay();
    currentStep = 0;
    renderStep(currentStep);
  });

  playBtn.addEventListener('click', () => {
    if (isPlaying) {
      stopPlay();
    } else {
      startPlay();
    }
  });

  function startPlay() {
    isPlaying = true;
    playBtn.textContent = 'Pause Loop';
    playBtn.classList.remove('btn-accent');
    playBtn.classList.add('btn-secondary');

    playInterval = setInterval(() => {
      if (currentStep < loopSteps.length - 1) {
        currentStep++;
        renderStep(currentStep);
      } else {
        stopPlay();
      }
    }, 2800);
  }

  function stopPlay() {
    isPlaying = false;
    if (playInterval) clearInterval(playInterval);
    playBtn.textContent = 'Auto Play Loop';
    playBtn.classList.add('btn-accent');
    playBtn.classList.remove('btn-secondary');
  }

  renderStep(0);
}

/* ==========================================================================
   11b. PART 7: AGENTIC AI CASE STUDY (SECTION 7.2)
   ========================================================================== */
const casePhases = [
  {
    role: 'TRIAGE & ORCHESTRATOR AGENT',
    roleColor: 'var(--accent-cyan)',
    title: 'Phase 1: Production Incident Detection & Context Assembly',
    timestamp: 'T+00:00:14 UTC',
    desc: 'Prometheus alerts trigger after GPU cluster node memory consumption exceeds 94% threshold during high-concurrency LLM inference. The Orchestrator agent correlates distributed trace IDs, queries Grafana metrics, and dispatches an investigation plan to sub-agents.',
    payload: `<span style="color:#64748b;">// Telemetry Signal &amp; Agent Goal Formulation</span><br>
<span style="color:#f43f5e;">ALERT: HighMemoryPressure</span> [Severity: Critical, Cluster: us-west-4a-h200]<br>
<span style="color:#38bdf8;">Goal:</span> Identify regression in KV-Cache eviction handler introduced within last 24h commits.<br>
<span style="color:#34d399;">Action:</span> Spawning Investigator Agent with read-only repository sandbox and AST tools.<br>
<span style="color:#fbbf24;">Tools Bound:</span> [prometheus_query, service_log_grep, git_repo_analyzer]`
  },
  {
    role: 'INVESTIGATOR AGENT',
    roleColor: 'var(--accent-purple)',
    title: 'Phase 2: Codebase Navigation, AST Call-Graph &amp; Git Bisect',
    timestamp: 'T+00:01:32 UTC',
    desc: 'The Investigator agent performs AST-grep across the 1.2M-LOC distributed runtime repository. Running git bisect against automated unit tests isolates commit #8f2a1c ("Optimize KV block reuse in FlashAttention-3 integration") as the offending memory leak source.',
    payload: `<span style="color:#64748b;">// Autonomous Tool Execution Trace</span><br>
<span style="color:#a78bfa;">$ tool_call:</span> git bisect start HEAD v4.12.0<br>
<span style="color:#38bdf8;">$ tool_call:</span> ast_grep --pattern 'class KVCacheManager { $$$ }'<br>
<span style="color:#34d399;">MATCH FOUND:</span> src/runtime/kv_cache_allocator.cpp:L248<br>
<span style="color:#f59e0b;">ROOT CAUSE:</span> Ref-count on PagedAttention memory arena not decremented on socket timeout exception.<br>
<span style="color:#a78bfa;">Dispatching:</span> Handing off localized bug specification &amp; reproduction test to Synthesis Agent.`
  },
  {
    role: 'SYNTHESIS &amp; PATCH AGENT',
    roleColor: 'var(--accent-emerald)',
    title: 'Phase 3: Sandboxed Code Modification &amp; Atomic Unified Diff',
    timestamp: 'T+00:02:45 UTC',
    desc: 'Operating in an isolated Docker container, the Synthesis Agent drafts an atomic hotfix patch implementing RAII-guarded lock release and exception-safe token buffer reclamation, adhering to C++20 memory order semantics.',
    payload: `<span style="color:#64748b;">// Synthesized Patch Diff (sandbox: docker://agent-sandbox-cpp:latest)</span><br>
<span style="color:#94a3b8;">--- a/src/runtime/kv_cache_allocator.cpp</span><br>
<span style="color:#94a3b8;">+++ b/src/runtime/kv_cache_allocator.cpp</span><br>
<span style="color:#f43f5e;">- void release_block(BlockId id) { if (!is_held(id)) return; ref_counts[id]--; }</span><br>
<span style="color:#34d399;">+ void release_block(BlockId id) noexcept {</span><br>
<span style="color:#34d399;">+   std::scoped_lock lock(arena_mutex_);</span><br>
<span style="color:#34d399;">+   if (ref_counts[id].fetch_sub(1, std::memory_order_acq_rel) == 1) { free_list_.push(id); }</span><br>
<span style="color:#34d399;">+ }</span><br>
<span style="color:#38bdf8;">Result:</span> Code compiles cleanly with 0 clang-tidy warnings. Handing to QA Critic Agent.`
  },
  {
    role: 'CRITIC / QA AUDITOR AGENT',
    roleColor: 'var(--accent-amber)',
    title: 'Phase 4: Automated CI Matrix Verification &amp; Canary Deployment',
    timestamp: 'T+00:04:18 UTC',
    desc: 'The Critic Agent generates a synthetic multi-threaded stress test simulating 10,000 socket drops. All 142 distributed cluster regression tests pass with zero memory leaks. A canary pull request is signed and staged for human-in-the-loop review.',
    payload: `<span style="color:#64748b;">// Verification Receipts &amp; Test Harness Execution</span><br>
<span style="color:#38bdf8;">$ tool_call:</span> pytest tests/distributed/test_kv_leak.py -v --workers=16<br>
<span style="color:#34d399;">TEST SUITE:</span> 142 Passed, 0 Failed, 0 Skipped [Time: 38.4s]<br>
<span style="color:#34d399;">VALGRIND / ASAN AUDIT:</span> 0 bytes lost in 0 allocations (Clean)<br>
<span style="color:#a78bfa;">CANARY STATUS:</span> Deployed to Pod 1 (canary pool: 128x H200 GPUs). Memory usage flat at 41%.<br>
<span style="color:#fbbf24;">PR #4182:</span> Created with full cryptographic audit trail &amp; regression receipts ready for approval.`
  }
];

let currentCasePhase = 0;
function showCasePhase(idx) {
  currentCasePhase = idx;
  const p = casePhases[idx];
  if (!p) return;

  for (let i = 0; i < 4; i++) {
    const pill = document.getElementById(`casePill${i}`);
    if (pill) pill.classList.toggle('active', i === idx);
  }

  const roleEl = document.getElementById('caseActiveRoleBadge');
  const titleEl = document.getElementById('casePhaseTitle');
  const timeEl = document.getElementById('caseTimestamp');
  const descEl = document.getElementById('casePhaseDesc');
  const payloadEl = document.getElementById('casePayloadBox');
  const prevBtn = document.getElementById('casePrevBtn');
  const nextBtn = document.getElementById('caseNextBtn');

  if (roleEl) {
    roleEl.textContent = p.role;
    roleEl.style.color = p.roleColor;
    roleEl.style.backgroundColor = idx === 0 ? 'rgba(56,189,248,0.15)' : (idx === 1 ? 'rgba(168,85,247,0.15)' : (idx === 2 ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)'));
  }
  if (titleEl) titleEl.textContent = p.title;
  if (timeEl) timeEl.textContent = p.timestamp;
  if (descEl) descEl.textContent = p.desc;
  if (payloadEl) payloadEl.innerHTML = p.payload;

  if (prevBtn) prevBtn.disabled = idx === 0;
  if (nextBtn) nextBtn.disabled = idx === casePhases.length - 1;
}

function initAgentCaseStudy() {
  const prevBtn = document.getElementById('casePrevBtn');
  const nextBtn = document.getElementById('caseNextBtn');

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (currentCasePhase > 0) showCasePhase(currentCasePhase - 1);
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      if (currentCasePhase < casePhases.length - 1) showCasePhase(currentCasePhase + 1);
    });
  }
  showCasePhase(0);
}

/* ==========================================================================
   12. PART 7: CHECKPOINT QUIZ ENGINE
   ========================================================================== */
function initQuizEngine() {
  const qNumEl = document.getElementById('quizQNum');
  const qTextEl = document.getElementById('quizQText');
  const optionsContainer = document.getElementById('quizOptionsContainer');
  const feedbackEl = document.getElementById('quizFeedback');
  const nextBtn = document.getElementById('quizNextBtn');
  const prevBtn = document.getElementById('quizPrevBtn');
  const scoreCounter = document.getElementById('quizScoreCounter');
  if (!qNumEl || !nextBtn) return;

  // 20 comprehensive questions covering all 9 curriculum parts and recent frontiers
  const questions = [
    {
      part: 'Part 1: Mathematical Foundations',
      text: 'In Bayes\' Theorem, if a disease has a 1% base rate, a test has a 90% sensitivity, and a 9% false positive rate, what is the posterior probability that a patient with a positive test actually has the disease?',
      options: [
        'Approximately 90% because test sensitivity is 90%',
        'Approximately 9.17% due to the low base rate (prior)',
        'Approximately 50% by symmetry',
        'Approximately 1% because the prior dominates completely'
      ],
      correct: 1,
      explanation: 'Applying Bayes\' Theorem: P(D|+) = (0.90 * 0.01) / [(0.90 * 0.01) + (0.09 * 0.99)] = 0.009 / (0.009 + 0.0891) ≈ 9.17%. This demonstrates the classic base-rate fallacy where false alarms outnumber true cases.'
    },
    {
      part: 'Part 1: Information Theory',
      text: 'Why is Cross-Entropy Loss the standard training objective for Next-Token Prediction in Large Language Models?',
      options: [
        'It minimizes compute FLOPs on matrix hardware',
        'It mathematically minimizes the KL divergence between the true token distribution and the model\'s predicted probability distribution',
        'It prevents any token from receiving a probability of zero',
        'It guarantees linear convergence in convex optimization'
      ],
      correct: 1,
      explanation: 'Minimizing cross-entropy H(P, Q) is equivalent to minimizing the Kullback-Leibler (KL) divergence D_KL(P || Q) from the model\'s distribution Q to the true empirical distribution P.'
    },
    {
      part: 'Part 2: Deep Neural Networks',
      text: 'What mathematical limitation of a single-layer perceptron did Minsky & Papert (1969) famously prove?',
      options: [
        'It cannot compute continuous real-valued functions',
        'It cannot learn linearly inseparable functions like XOR with a single hyperplane',
        'It requires infinite epochs to converge on linearly separable data',
        'It cannot be trained using gradient descent'
      ],
      correct: 1,
      explanation: 'A single perceptron forms a linear decision boundary (hyperplane). Because XOR outputs are 1 on opposite diagonals and 0 on the other, no single line can separate them without at least one hidden layer.'
    },
    {
      part: 'Part 2: Backpropagation',
      text: 'What mathematical rule enables backpropagation to compute gradients through dozens of stacked neural network layers?',
      options: [
        'Taylor Series Expansion',
        'The Multivariate Chain Rule of Calculus',
        'Bayes\' Rule of Conditional Probability',
        'Euler\'s Identity'
      ],
      correct: 1,
      explanation: 'Backpropagation is a dynamic programming implementation of the multivariate Chain Rule: ∂L/∂W^[l] = (∂L/∂a^[l]) · (∂a^[l]/∂z^[l]) · (∂z^[l]/∂W^[l]).'
    },
    {
      part: 'Part 2: Modern Optimization',
      text: 'How does the Adam optimizer improve upon standard Stochastic Gradient Descent (SGD) during deep network training?',
      options: [
        'By computing exact second-order Hessian matrices at each step',
        'By combining exponentially decaying first-moment momentum with adaptive second-moment variance scaling',
        'By eliminating the need for learning rates entirely',
        'By restricting weight updates to integer values'
      ],
      correct: 1,
      explanation: 'Adam (Kingma & Ba, 2014) maintains moving averages of both past gradients (momentum) and squared gradients (uncentered variance), providing adaptive per-parameter learning rates resilient to sparse or noisy gradients.'
    },
    {
      part: 'Part 3: Computer Vision (CNNs)',
      text: 'What core inductive biases allow Convolutional Neural Networks (CNNs) to process high-resolution images far more efficiently than dense MLPs?',
      options: [
        'Recurrent temporal cycles and memory gates',
        'Local receptive fields and spatial weight sharing, granting translation equivariance and preventing parameter explosion',
        'Self-attention matrices computed across every pixel pair',
        'Purely linear non-parametric filtering'
      ],
      correct: 1,
      explanation: 'CNNs slide small parameter kernels across spatial dimensions (weight sharing) and restrict connections to local pixel neighborhoods, dramatically reducing parameters from O(H²W²) to small constant kernel sizes while preserving translation equivariance.'
    },
    {
      part: 'Part 3: Sequential Architectures',
      text: 'What structural bottleneck in Recurrent Neural Networks (RNNs) does the Transformer\'s Attention mechanism eliminate?',
      options: [
        'Overfitting on training data',
        'Sequential dependency that prevented parallelization across the sequence length',
        'The need for activation functions',
        'The requirement of embedding tokens into vectors'
      ],
      correct: 1,
      explanation: 'RNNs require step-by-step computation (h_t depends on h_{t-1}), creating an O(N) sequential bottleneck that cannot be parallelized. Transformers process all tokens simultaneously in O(1) sequential steps.'
    },
    {
      part: 'Part 4: Reinforcement Learning Fundamentals',
      text: 'In Reinforcement Learning, what recursive relationship expresses the value of a state as the expected immediate reward plus the discounted future value of downstream states?',
      options: [
        'The Shannon Entropy Equation',
        'The Bellman Optimality Equation',
        'The Universal Approximation Theorem',
        'The Backpropagation Chain Rule'
      ],
      correct: 1,
      explanation: 'The Bellman Equation V(s) = E[r_t + γ V(s_{t+1}) | s_t = s] provides the foundational recursive framework for value function estimation and dynamic programming in Markov Decision Processes.'
    },
    {
      part: 'Part 4: Supervised Learning vs. Reinforcement Learning',
      text: 'What fundamental challenge distinguishes Reinforcement Learning from standard Supervised Learning?',
      options: [
        'RL models cannot use gradient descent',
        'The Credit Assignment Problem: actions only yield delayed scalar rewards rather than immediate ground-truth target vectors',
        'Supervised learning cannot be applied to classification',
        'RL requires all data to be labeled manually by human annotators before training'
      ],
      correct: 1,
      explanation: 'In supervised learning, each input has an immediate correct label. In RL, an agent must balance exploration vs. exploitation under delayed, sparse rewards where credit must be assigned across dozens of preceding sequential decisions.'
    },
    {
      part: 'Part 4: Reinforcement Learning Frontiers',
      text: 'How did AlphaGo Zero (Silver et al., 2017) achieve superhuman performance in the game of Go without human expert training games?',
      options: [
        'By brute-force lookup of all possible 10^170 board configurations',
        'Through pure tabula-rasa self-play combining Monte Carlo Tree Search (MCTS) with deep neural policy/value guidance',
        'By fine-tuning a pre-trained vision-language model',
        'By utilizing rule-based handcrafted expert heuristics'
      ],
      correct: 1,
      explanation: 'AlphaGo Zero learned entirely tabula rasa through self-play reinforcement learning, iteratively improving its policy and value networks using MCTS search probabilities without any human games or domain knowledge beyond the rules.'
    },
    {
      part: 'Part 5: Attention Mechanism',
      text: 'In the Scaled Dot-Product Attention formula Attention(Q, K, V) = Softmax(QK^T / √d_k)V, why is the division by √d_k essential?',
      options: [
        'To prevent dot products from growing excessively large, which pushes the softmax into regions with vanishing gradients',
        'To convert dot products into Euclidean distances',
        'To ensure the attention weights sum to zero',
        'To reduce the memory complexity of the matrix multiplication'
      ],
      correct: 0,
      explanation: 'As dimension d_k grows, dot products grow large in magnitude, driving softmax into regions with extremely small gradients. Dividing by √d_k stabilizes the variance to 1.'
    },
    {
      part: 'Part 5: Generative AI & Scaling',
      text: 'According to the Chinchilla Scaling Laws (Hoffmann et al., 2022), what is the compute-optimal ratio between training tokens and model parameters?',
      options: [
        '1 token per parameter',
        'Approximately 20 tokens per parameter',
        '1,000 tokens per parameter',
        'Model size should scale with the square of dataset tokens'
      ],
      correct: 1,
      explanation: 'Chinchilla demonstrated that model parameters and training tokens should be scaled in equal proportion: approximately 20 tokens per parameter for compute-optimal training.'
    },
    {
      part: 'Part 5: Post-Training & Alignment',
      text: 'What is the "Alignment Tax" frequently observed after Reinforcement Learning from Human Feedback (RLHF)?',
      options: [
        'The cloud billing cost of hiring human annotators',
        'The slight degradation in raw capability, creative diversity, or code accuracy that can occur when safety penalties constrain a model',
        'The government regulatory fee for deploying frontier AI',
        'The GPU memory overhead of loading reward model weights'
      ],
      correct: 1,
      explanation: 'The Alignment Tax describes the empirical phenomenon where models heavily aligned for conversational safety and deference sometimes show slight performance regressions in raw mathematics, coding, or diverse completions.'
    },
    {
      part: 'Part 6: Hardware & Clusters',
      text: 'Why do GPUs dramatically outperform CPUs for training large neural networks?',
      options: [
        'GPUs run at higher single-core clock frequencies (GHz)',
        'GPUs feature thousands of smaller SIMD/SIMT arithmetic logic units (ALUs) optimized for parallel matrix multiplications',
        'GPUs have larger L1/L2 cache sizes than CPUs',
        'GPUs do not require cooling infrastructure'
      ],
      correct: 1,
      explanation: 'CPUs prioritize low latency for sequential code with complex branching; GPUs prioritize massive throughput for parallel data pipelines with thousands of ALUs executing identical operations simultaneously.'
    },
    {
      part: 'Part 6: Cluster Networking & NVIDIA H200',
      text: 'As an AI compute cluster scales from an 8-GPU server to an entire 3,072-GPU Pod, what happens to inter-GPU communication bandwidth in an NVIDIA H200 cluster?',
      options: [
        'It increases exponentially due to distributed caching',
        'It collapses by roughly 18x to ~96x (from 4,800 GB/s on-package HBM3e / 900 GB/s NVLink down to ~50 GB/s over fabric switches)',
        'It remains constant across all hierarchy boundaries',
        'Bandwidth becomes infinite using optical quantum entanglement'
      ],
      correct: 1,
      explanation: 'Physical wiring distance and network switch layers create a steep physical hierarchy: on-package HBM3e runs at 4,800 GB/s (4.8 TB/s), NVLink intra-server at 900 GB/s, and pod/cluster InfiniBand drops to ~50 GB/s — a ~96x total bandwidth collapse.'
    },
    {
      part: 'Part 7: Agentic AI Systems',
      text: 'What fundamental capability distinguishes an Agentic AI system from a standard conversational Large Language Model?',
      options: [
        'Agentic AI is always trained on more parameters',
        'Agentic AI operates in a goal-driven loop with state memory, dynamic planning, tool calling, and self-reflection',
        'Agentic AI uses recurrent neural networks instead of transformers',
        'Agentic AI does not use prompt engineering'
      ],
      correct: 1,
      explanation: 'Standard LLMs follow a passive input→output pattern; Agentic AI wraps the model in a closed loop (Plan-Execute-Evaluate-Reflect) equipped with external tools, state memory, and multi-step reasoning.'
    },
    {
      part: 'Part 7: Russell & Norvig Agency',
      text: 'What does the classical AI acronym PEAS stand for when designing an autonomous agent?',
      options: [
        'Parallel Execution, Allocation, and Storage',
        'Performance measure, Environment, Actuators, Sensors',
        'Perception, Evaluation, Attention, Superposition',
        'Probability, Entropy, Alignment, Scaling'
      ],
      correct: 1,
      explanation: 'PEAS stands for Performance measure, Environment, Actuators, and Sensors — the foundational framework formulated by Russell & Norvig for rational agent specification.'
    },
    {
      part: 'Part 7: Production Multi-Agent Swarms',
      text: 'In an autonomous enterprise software engineering swarm (such as incident SRE remediation), what role does the Critic / QA Auditor agent perform?',
      options: [
        'It writes the original production application code',
        'It triages customer support tickets and manages cloud billing',
        'It generates reproduction test matrices, runs regression suites and memory profilers, and signs off on pull requests before deployment',
        'It deploys unreviewed patches directly to production without testing'
      ],
      correct: 2,
      explanation: 'Role specialization separates patch synthesis from rigorous validation: the Critic/QA agent generates adversarial reproduction tests and runs ASAN/Valgrind profiling in sandboxed containers to ensure patches introduce zero regressions.'
    },
    {
      part: 'Part 8: Physical AI & Humanoid Robotics',
      text: 'What dynamic balance criterion must a whole-body bipedal humanoid robot satisfy during dynamic locomotion?',
      options: [
        'The center of mass must remain statically positioned between both feet at all times',
        'The Zero Moment Point (ZMP) — where horizontal tipping moments vanish — must remain strictly within the convex hull of the support foot polygon',
        'The robot must keep its knees locked to maximize mechanical advantage',
        'Joint torque commands only need to be computed at 1 Hz'
      ],
      correct: 1,
      explanation: 'In dynamic bipedal walking, balance requires that the Zero Moment Point (ZMP) remains inside the foot support polygon. Mid-level Model Predictive Control (MPC) plans Center-of-Mass trajectories to enforce this condition at 50–100Hz while 1,000Hz motor controllers track joint torques.'
    },
    {
      part: 'Part 9: The AI Value Paradigm Shift',
      text: 'Why does the curriculum argue that treating generative AI as a "1-to-1 search lookup engine" is an economic mistake?',
      options: [
        'Because search engines require no electricity',
        'Because single-turn query lookups are commoditized and ephemeral, whereas compounding enterprise value resides in multi-agent workflows, multimodal creation, persistent memory, and reusable systems',
        'Because AI models cannot index web pages',
        'Because traditional databases have higher compute latency'
      ],
      correct: 1,
      explanation: 'Single-turn prompt lookup provides zero competitive moat and static knowledge cutoffs. Transformative enterprise value compounds when AI acts as an autonomous orchestration engine with domain RAG, multi-agent workflows, and persistent memory assets.'
    }
  ];

  let currentQ = 0;
  let userAnswers = new Array(questions.length).fill(null);

  function renderQuestion(idx) {
    const q = questions[idx];
    qNumEl.textContent = `${q.part} • Question ${idx + 1} of ${questions.length}`;
    qTextEl.textContent = q.text;

    optionsContainer.innerHTML = '';
    feedbackEl.className = 'quiz-feedback-box';
    feedbackEl.style.display = 'none';

    q.options.forEach((optText, optIdx) => {
      const btn = document.createElement('button');
      btn.className = 'quiz-opt-btn';
      const letter = String.fromCharCode(65 + optIdx);
      btn.innerHTML = `<span class="quiz-opt-letter">${letter}.</span> <span>${optText}</span>`;

      if (userAnswers[idx] !== null) {
        if (optIdx === q.correct) {
          btn.classList.add('correct');
        } else if (optIdx === userAnswers[idx]) {
          btn.classList.add('incorrect');
        }
      }

      btn.addEventListener('click', () => {
        if (userAnswers[idx] !== null) return; // already answered
        userAnswers[idx] = optIdx;
        if (optIdx === q.correct) {
          btn.classList.add('correct');
          feedbackEl.className = 'quiz-feedback-box correct';
          feedbackEl.innerHTML = `<strong>✓ Correct!</strong> ${q.explanation}`;
        } else {
          btn.classList.add('incorrect');
          feedbackEl.className = 'quiz-feedback-box incorrect';
          feedbackEl.innerHTML = `<strong>✗ Incorrect.</strong> The correct answer is <strong>${String.fromCharCode(65 + q.correct)}</strong>.<br>${q.explanation}`;
          // Also highlight correct button
          optionsContainer.children[q.correct].classList.add('correct');
        }
        feedbackEl.style.display = 'block';
        updateScore();
      });

      optionsContainer.appendChild(btn);
    });

    if (userAnswers[idx] !== null) {
      feedbackEl.style.display = 'block';
      if (userAnswers[idx] === q.correct) {
        feedbackEl.className = 'quiz-feedback-box correct';
        feedbackEl.innerHTML = `<strong>✓ Correct!</strong> ${q.explanation}`;
      } else {
        feedbackEl.className = 'quiz-feedback-box incorrect';
        feedbackEl.innerHTML = `<strong>✗ Incorrect.</strong> The correct answer is <strong>${String.fromCharCode(65 + q.correct)}</strong>.<br>${q.explanation}`;
      }
    }

    prevBtn.disabled = (idx === 0);
    nextBtn.disabled = (idx === questions.length - 1);
  }

  function updateScore() {
    let score = 0;
    let answered = 0;
    userAnswers.forEach((ans, i) => {
      if (ans !== null) {
        answered++;
        if (ans === questions[i].correct) score++;
      }
    });
    scoreCounter.textContent = `Score: ${score} / ${questions.length} (${answered} answered)`;
  }

  nextBtn.addEventListener('click', () => {
    if (currentQ < questions.length - 1) {
      currentQ++;
      renderQuestion(currentQ);
    }
  });

  prevBtn.addEventListener('click', () => {
    if (currentQ > 0) {
      currentQ--;
      renderQuestion(currentQ);
    }
  });

  renderQuestion(0);
  updateScore();
}

/* ==========================================================================
   13. PART 9 / APPENDIX A: SEARCHABLE GLOSSARY
   ========================================================================== */
function initGlossary() {
  const input = document.getElementById('glossarySearchInput');
  const grid = document.getElementById('glossaryGrid');
  if (!input || !grid) return;

  const glossaryData = [
    { term: 'Perceptron', def: 'The fundamental atomic artificial neuron model that computes a weighted sum of inputs plus a bias term, passed through an activation function: y = f(w^T x + b).' },
    { term: 'Universal Approximation Theorem', def: 'Mathematical theorem proving that a feedforward network with a single hidden layer and non-linear activation can approximate any continuous function on compact subsets.' },
    { term: 'Backpropagation', def: 'The foundational algorithm for computing partial derivatives of the loss function with respect to all network weights using the multivariate chain rule in reverse mode.' },
    { term: 'Cross-Entropy Loss', def: 'An information-theoretic loss function quantifying the divergence between the true probability distribution and the model\'s predicted probability distribution, equivalent to minimizing KL divergence.' },
    { term: 'Adam Optimizer', def: 'Adaptive moment estimation optimizer combining exponentially decaying first-moment momentum with second-moment uncentered variance scaling for robust per-parameter learning rates.' },
    { term: 'Vanishing Gradient Problem', def: 'Exponential decay of gradient magnitudes as backpropagation flows backward through deep recurrent or feedforward layers, mitigated by ReLU activations and residual skip connections.' },
    { term: 'Convolutional Kernel', def: 'A localized spatial matrix with tied/shared weights slid across multidimensional inputs to extract translation-equivariant features while drastically preventing parameter explosion.' },
    { term: 'LSTM Cell', def: 'Long Short-Term Memory recurrent unit using Constant Error Carousels governed by multiplicative Forget, Input, and Output gates to preserve gradients across long temporal horizons.' },
    { term: 'Self-Attention', def: 'A mechanism enabling tokens in a sequence to dynamically attend to and weight relationships with all other tokens using Query, Key, and Value projections.' },
    { term: 'Scaled Dot-Product Attention', def: 'Attention formulation Attention(Q,K,V) = Softmax(QK^T / √d_k)V; dividing by √d_k stabilizes variance to 1 and prevents softmax gradients from vanishing.' },
    { term: 'Transformer', def: 'The dominant neural network architecture based entirely on self-attention mechanisms and feed-forward layers, eliminating recurrent sequential constraints for O(1) sequential depth.' },
    { term: 'Markov Decision Process (MDP)', def: 'The formal mathematical 5-tuple (S, A, P, R, γ) modeling sequential decision-making where state transitions satisfy the Markov property.' },
    { term: 'Bellman Equation', def: 'Recursive optimality equation decomposing a state\'s value into the expected immediate reward plus the discounted expected downstream value: V(s) = E[r + γ V(s\')].' },
    { term: 'Exploration vs. Exploitation', def: 'The fundamental reinforcement learning trade-off between trying novel actions to discover better rewards and executing the current best known policy.' },
    { term: 'Deep Q-Network (DQN)', def: 'Deep reinforcement learning algorithm approximating the optimal state-action value function Q*(s, a) using deep neural networks stabilized by experience replay and target networks.' },
    { term: 'Policy Gradient & PPO', def: 'Algorithms that optimize policy network parameters directly via gradient ascent; Proximal Policy Optimization (PPO) uses clipped surrogate objectives to prevent destructive updates.' },
    { term: 'Scaling Laws (Chinchilla)', def: 'Empirical power laws (Hoffmann et al., 2022) proving that model parameters and training dataset tokens should scale in equal proportion (~20 tokens per parameter) for compute optimality.' },
    { term: 'RLHF', def: 'Reinforcement Learning from Human Feedback: post-training alignment pipeline using human preference ratings to train a reward model, which guides policy optimization via PPO.' },
    { term: 'Direct Preference Optimization (DPO)', def: 'Closed-form alignment algorithm optimizing language model weights directly on paired human preference completions without needing an explicit reward model.' },
    { term: 'Alignment Tax', def: 'The empirical trade-off where safety-alignment and deference constraints slightly degrade raw performance on challenging technical, mathematical, or creative benchmarks.' },
    { term: 'Tensor Core', def: 'Specialized hardware execution units designed specifically to perform mixed-precision matrix multiply-accumulate operations (e.g. FP8, FP16) in a single clock cycle.' },
    { term: 'HBM3e (High Bandwidth Memory)', def: '3D-stacked memory architecture providing ultra-high capacity and bandwidth (141GB capacity at 4,800 GB/s on NVIDIA H200) directly over a silicon interposer.' },
    { term: 'NVLink 4.0', def: 'NVIDIA\'s proprietary high-speed interconnect protocol enabling direct, high-bandwidth GPU-to-GPU communications (900 GB/s bidirectional per GPU).' },
    { term: 'CoWoS-S Silicon Interposer', def: 'Chip-on-Wafer-on-Substrate packaging technology providing dense microscopic wiring connecting the GPU compute die directly to adjacent 3D HBM stacks.' },
    { term: 'GPU SuperPOD / Hyperscale AI Factory', def: 'Cluster computing architecture aggregating thousands of GPUs (e.g., 3,072 to 24,576 H200s delivering 24+ ExaFLOPs) through non-blocking optical InfiniBand fabric networks.' },
    { term: 'Agentic AI', def: 'Autonomous cognitive systems capable of active perception, multi-step planning, tool use, environment interaction, and self-reflection in pursuit of high-level goals.' },
    { term: 'PEAS Framework', def: 'Russell and Norvig\'s standard specification for rational agents: Performance measure, Environment, Actuators, and Sensors.' },
    { term: 'ReAct Pattern', def: 'Synergizing Reasoning and Acting in LLMs by interleaving internal chain-of-thought generation with external sandbox tool execution.' },
    { term: 'Multi-Agent Swarm', def: 'A collective of specialized autonomous agents (e.g. Orchestrator, Investigator, Patch Synthesizer, QA Auditor) collaborating via asynchronous DAG workflows to solve enterprise engineering tasks.' },
    { term: 'Tool Calling & Reflection', def: 'The mechanism allowing LLMs to emit structured API invocations, inspect return values, and verbally critique their own intermediate outputs to correct errors.' },
    { term: 'Vision-Language-Action (VLA)', def: 'Multimodal foundation models (e.g. RT-2, OpenVLA) co-trained on web-scale images, text, and robot demonstrations that directly output continuous motor control actions.' },
    { term: 'Model Predictive Control (MPC)', def: 'Dynamic online control optimization solving finite-horizon trajectory problems at 50–100Hz to calculate optimal center-of-mass positions and footstep placements.' },
    { term: 'Zero Moment Point (ZMP)', def: 'The dynamic balance criterion for bipedal humanoid walking: the ground point where horizontal inertial and gravitational moments sum to zero, which must lie within the foot support polygon.' },
    { term: 'Shannon Entropy', def: 'The foundational mathematical measure of information content and uncertainty H(X) = -∑ p(x) log2 p(x) introduced by Claude Shannon (1948), establishing the formal basis for modern cross-entropy loss.' },
    { term: 'Bayesian Updating', def: 'The formal calculus of scientific belief revision computing posterior probabilities P(θ|D) ∝ P(D|θ) P(θ) from observed data and prior distributions, governing probabilistic inference.' },
    { term: 'Polynomial Recurrent Neural Network (PRNN)', def: 'Non-linear recurrent architecture developed by Haykin & Li (1995) introducing polynomial Kronecker feedback with recursive internal states to capture complex non-linear temporal dynamics.' },
    { term: 'Autonomous Coding Agent', def: 'An autonomous agentic system (e.g. Gemini Antigravity, Claude Code, Cursor) that executes continuous software engineering loops: inspecting filesystems, planning changes, editing code, running test suites, and handling git version control.' },
    { term: 'SWE-bench Benchmark', def: 'The premier academic evaluation harness (Princeton University) measuring the capability of autonomous coding agents to resolve real-world GitHub issues by editing codebases and passing hidden unit test suites.' },
    { term: 'Explore-Plan-Implement-Verify', def: 'The disciplined 4-stage engineering protocol for autonomous software development: read-only discovery, architectural planning, surgical implementation, and automated test-driven verification.' },
    { term: 'Git Worktree Concurrency', def: 'An advanced multi-agent development pattern using isolated git linked worktrees, allowing parallel AI agents to develop, build, and test features concurrently without file locks or branch collisions.' },
    { term: "Moravec's Paradox", def: 'The historic AI discovery that high-level abstract logic and game-playing require very little compute, whereas basic sensorimotor perception, physical balance, and robotic dexterity require vast compute.' },
    { term: 'Sim-to-Real Transfer', def: 'The process of training reinforcement learning policies inside massive parallel GPU physics simulators (Isaac Sim) with domain randomization, then zero-shot deploying them to physical hardware.' },
    { term: 'High-Order Value Creation', def: 'The strategic shift from commoditized, ephemeral 1-to-1 prompt lookups to compounding enterprise assets: autonomous workflows, multimodal synthesis, and proprietary cognitive systems.' }
  ];

  function render(query = '') {
    const q = query.toLowerCase().trim();
    grid.innerHTML = '';
    const filtered = glossaryData.filter(item =>
      item.term.toLowerCase().includes(q) || item.def.toLowerCase().includes(q)
    );

    if (filtered.length === 0) {
      grid.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding:2rem; color:var(--text-muted);">No matching glossary terms found for "${query}".</div>`;
      return;
    }

    filtered.forEach(item => {
      const card = document.createElement('div');
      card.className = 'glossary-card';
      card.innerHTML = `
        <div class="glossary-term">${item.term}</div>
        <div class="glossary-def">${item.def}</div>
      `;
      grid.appendChild(card);
    });
  }

  input.addEventListener('input', (e) => {
    render(e.target.value);
  });

  render();
}


/* ==========================================================================
   14. PART 10: ACADEMIC REFERENCES ARCHIVE FILTER
   ========================================================================== */
function initReferencesArchive() {
  const filterBtns = document.querySelectorAll('.ref-filter-btn');
  const cards = document.querySelectorAll('.ref-card');
  if (!filterBtns.length || !cards.length) return;

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.getAttribute('data-filter');
      cards.forEach(card => {
        const cat = card.getAttribute('data-category') || '';
        if (filter === 'all' || cat === filter || cat.split(/\s+/).includes(filter)) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });
}


/* ==========================================================================
   15. SPOTLIGHT QUICK SEARCH MODAL (CTRL + K)
   ========================================================================== */
function initSpotlightSearch() {
  const modal = document.getElementById('spotlightModal');
  const triggerBtn = document.getElementById('spotlightTriggerBtn');
  const input = document.getElementById('spotlightSearchInput');
  const resultsContainer = document.getElementById('spotlightResultsContainer');
  if (!modal || !input || !resultsContainer) return;

  const searchableIndex = [
    { title: 'Bayes Theorem & Base-Rate Fallacy Calculator', badge: 'Part 1', target: '#bayesCalculatorLab', desc: 'Prior prevalence P(C)=1%, test sensitivity 90%, posterior updating P(C|+)=9.17%' },
    { title: 'Linear & Logistic Regression Lab', badge: 'Part 1', target: '#part1', desc: 'Continuous prediction y = wx + b, Sigmoid activation, MSE loss calculation' },
    { title: 'Shannon Information Theory & Entropy', badge: 'Part 1', target: '#part1', desc: 'Entropy H(X) = -sum p log2(p), Cross-Entropy loss, KL Divergence' },
    { title: 'Perceptron & XOR Non-Linear Boundary', badge: 'Part 2', target: '#part2', desc: 'Single neuron linear limitation, Minsky & Papert proof, 2-layer MLP solution' },
    { title: 'Multi-Layer Perceptron (MLP) Architecture', badge: 'Part 2', target: '#part2-mlp', desc: 'Matrix forward prop z^[l] = W^[l]a^[l-1]+b^[l], Universal Approximation Theorem' },
    { title: 'Activation Functions: GELU, ReLU, Sigmoid', badge: 'Part 2', target: '#part2', desc: 'Non-linear transformations, derivative curves, vanishing gradient mitigation' },
    { title: 'How Deep Networks Learn: 5-Step Universal Cycle (Supervised Learning)', badge: 'Part 2', target: '#part2-learning-cycle', desc: 'Forward prop, Loss, Backpropagation chain rule, Gradient descent, Epochs' },
    { title: 'Convolutional Networks (CNN) & Feature Map Extraction', badge: 'Part 3', target: '#part3-cnn', desc: 'Local receptive fields, 2D convolution (I * W), spatial weight sharing, feature maps, AlexNet to ResNet' },
    { title: 'Sequential Memory: RNNs & LSTMs', badge: 'Part 3', target: '#part3-rnn-lstm', desc: 'BPTT gradient decay, single-tanh RNN cell, Constant Error Carousel, Forget/Input/Output gates' },
    { title: 'Reinforcement Learning: Agent-Environment Loop & MDP', badge: 'Part 4', target: '#part4-rl-principles', desc: 'State St, Action At, Reward Rt, Bellman recursive optimality, exploration vs exploitation' },
    { title: 'DeepMind Landmark RL Revolution: AlphaGo to AlphaFold', badge: 'Part 4', target: '#part4-rl-principles', desc: 'AlphaGo (2016), AlphaZero (2017), MuZero (2019), and AlphaFold (2020s)' },
    { title: 'Supervised Learning vs. Reinforcement Learning', badge: 'Part 4', target: '#part4-sl-vs-rl', desc: 'Direct labels vs delayed scalar rewards, static data vs online interaction' },
    { title: 'Applications of Reinforcement Learning', badge: 'Part 4', target: '#part4-rl-applications', desc: 'AlphaZero, robot dexterity, RLHF alignment, real-time bidding, data center cooling' },
    { title: 'Attention Mechanism & Transformer Model', badge: 'Part 5', target: '#part5-attention', desc: 'Query-Key-Value routing, sqrt(d_k) variance scaling, Multi-Head Attention, RoPE' },
    { title: 'Chinchilla Empirical Scaling Laws', badge: 'Part 5', target: '#part5-scaling', desc: 'Compute-optimal tokens D ≈ 20*N, 6ND FLOPs, frontier 18T tokens scale' },
    { title: 'NVIDIA H200 GPU Microarchitecture', badge: 'Part 6', target: '#part6', desc: 'Streaming Multiprocessors, 4th-Gen FP8 Tensor Cores, 141GB HBM3e @ 4,800 GB/s' },
    { title: 'H200 AI Supercluster Scale: Chip to Data Center', badge: 'Part 6', target: '#part6', desc: '5-tier hierarchy, 24,576 H200 GPUs, 24+ ExaFLOPs, ~96x bandwidth drop waterfall' },
    { title: 'Autonomous Agentic AI & The Agent Loop', badge: 'Part 7', target: '#part7', desc: 'Plan -> Execute -> Evaluate -> Reflect feedback engine, tool sandbox' },
    { title: 'Agentic AI Production Case Study: Enterprise SRE', badge: 'Part 7', target: '#agentCaseStudyContainer', desc: 'Multi-agent software engineering & incident remediation: Orchestrator, Investigator, Synthesizer, QA Critic' },
    { title: 'PEAS Rational Agency Framework', badge: 'Part 7', target: '#part7', desc: 'Performance measure, Environment, Actuators, Sensors (Russell & Norvig)' },
    { title: 'Coding Agents & Autonomous Software Engineering', badge: 'Part 8', target: '#part8', desc: 'Four generations of AI programming, tool invocation loops, and background agents' },
    { title: 'Gemini 2.5/3 Agentic Engine & Antigravity Tri-Surface Architecture', badge: 'Part 8', target: '#part8-gemini', desc: '1M+ token context, JSON Schema native tool calling, Editor-Terminal-Browser execution' },
    { title: 'Disciplined 4-Stage Workflow: Explore, Plan, Implement, Verify', badge: 'Part 8', target: '#part8-workflow', desc: 'Risk minimization formula, TDD fail-first reproduction, zero placeholders, and automated quality gates' },
    { title: 'Layered Safety Controls & Execution Sandboxing', badge: 'Part 8', target: '#part8-safety', desc: 'Approval gates, allow/deny lists, filesystem path confinement, snapshot rollbacks, git net' },
    { title: 'Parallel Agents in Git Worktrees', badge: 'Part 8', target: '#part8-worktrees', desc: 'Multi-agent horizontal scaling without file lock conflicts or branch collisions' },
    { title: 'Physical AI & Embodied Intelligence', badge: 'Part 9', target: '#part9', desc: 'Embodied robotics, multimodal sensory grounding, VLA models, sub-10ms control' },
    { title: 'Physical AI for Humanoid Robotics & Whole-Body Control', badge: 'Part 9', target: '#part9', desc: '28–50+ DoF, Zero Moment Point (ZMP), 1,000Hz torque control, Sim-to-Real RL in Isaac Gym' },
    { title: 'Vision-Language-Action (VLA) Models', badge: 'Part 9', target: '#part9', desc: 'RT-1, RT-2, OpenVLA, Octo: bridging language intent to discrete motor tokens' },
    { title: 'Robotic Kinematics & Sim-to-Real Lab', badge: 'Part 9', target: '#part9', desc: 'Forward kinematics, 2x2 Jacobian matrix, Isaac Sim physics transfer' },
    { title: 'Authoritative Key Terms Glossary', badge: 'Glossary', target: '#glossarySection', desc: 'Searchable definitions: Backprop, Attention, RLHF, NVLink, PEAS, etc.' },
    { title: 'Academic Reference Archive & Bibliography', badge: 'Part 10', target: '#referencesSection', desc: '32 landmark papers: Shannon 1948, Vaswani 2017, Chinchilla 2022, etc.' }
  ];

  function openSpotlight() {
    modal.classList.add('active');
    input.value = '';
    renderResults('');
    setTimeout(() => input.focus(), 50);
  }

  function closeSpotlight() {
    modal.classList.remove('active');
  }

  function renderResults(q = '') {
    const query = q.toLowerCase().trim();
    resultsContainer.innerHTML = '';
    const filtered = query ? searchableIndex.filter(item =>
      item.title.toLowerCase().includes(query) ||
      item.desc.toLowerCase().includes(query) ||
      item.badge.toLowerCase().includes(query)
    ) : searchableIndex.slice(0, 8);

    if (!filtered.length) {
      resultsContainer.innerHTML = '<div style="padding:1.5rem; text-align:center; color:var(--text-muted); font-size:0.9rem;">No matching curriculum topics found.</div>';
      return;
    }

    filtered.forEach(item => {
      const a = document.createElement('a');
      a.className = 'spotlight-item';
      a.href = item.target;
      a.innerHTML = `
        <div class="spotlight-item-text">
          <strong>${item.title}</strong>
          <span>${item.desc}</span>
        </div>
        <span class="spotlight-item-badge">${item.badge}</span>
      `;
      a.addEventListener('click', () => {
        closeSpotlight();
      });
      resultsContainer.appendChild(a);
    });
  }

  if (triggerBtn) {
    triggerBtn.addEventListener('click', openSpotlight);
  }

  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (modal.classList.contains('active')) closeSpotlight();
      else openSpotlight();
    }
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeSpotlight();
    }
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeSpotlight();
  });

  input.addEventListener('input', (e) => {
    renderResults(e.target.value);
  });
}

/* ==========================================================================
   16. ENLARGED HIGH-RES DIAGRAM LIGHTBOX
   ========================================================================== */
function initLightbox() {
  const modal = document.getElementById('lightboxModal');
  const lightboxImg = document.getElementById('lightboxImg');
  const captionEl = document.getElementById('lightboxCaption');
  const closeBtn = document.getElementById('lightboxCloseBtn');
  if (!modal || !lightboxImg || !captionEl) return;

  const figures = document.querySelectorAll('.figure-container');
  figures.forEach(fig => {
    const img = fig.querySelector('img');
    const cap = fig.querySelector('.figure-caption');
    if (!img) return;

    img.style.cursor = 'zoom-in';
    img.title = 'Click to enlarge diagram';

    img.addEventListener('click', () => {
      lightboxImg.src = img.src;
      captionEl.innerHTML = cap ? cap.innerHTML : img.alt;
      modal.classList.add('active');
    });
  });

  function closeLightbox() {
    modal.classList.remove('active');
  }

  if (closeBtn) closeBtn.addEventListener('click', closeLightbox);
  modal.addEventListener('click', (e) => {
    if (e.target === modal || e.target.classList.contains('lightbox-desc-bar')) {
      closeLightbox();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeLightbox();
    }
  });
}



/* ==========================================================================
   18. PART 7: PHYSICAL AI ROBOTIC ARM KINEMATICS LAB
   ========================================================================== */
function initKinematicsLab() {
  const canvas = document.getElementById('kinematicsCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const theta1Slider = document.getElementById('armTheta1');
  const theta2Slider = document.getElementById('armTheta2');
  if (!theta1Slider || !theta2Slider) return;

  const theta1Val = document.getElementById('armTheta1Val');
  const theta2Val = document.getElementById('armTheta2Val');
  const endXVal = document.getElementById('armEndXVal');
  const endYVal = document.getElementById('armEndYVal');
  const jacobianVal = document.getElementById('armJacobianVal');
  const safetyStatus = document.getElementById('armSafetyStatus');

  const l1 = 130; // Link 1 length
  const l2 = 100; // Link 2 length

  // Obstacle box for physical safety compliance
  const obstacle = { x: 80, y: -90, w: 70, h: 70 };

  function render() {
    const deg1 = parseFloat(theta1Slider.value);
    const deg2 = parseFloat(theta2Slider.value);
    theta1Val.textContent = deg1.toFixed(0) + '°';
    theta2Val.textContent = deg2.toFixed(0) + '°';

    const rad1 = (deg1 * Math.PI) / 180;
    const rad2 = (deg2 * Math.PI) / 180;

    // Forward Kinematics (FK)
    // Base origin at canvas center-bottom
    const x1 = l1 * Math.cos(rad1);
    const y1 = l1 * Math.sin(rad1);

    const x2 = x1 + l2 * Math.cos(rad1 + rad2);
    const y2 = y1 + l2 * Math.sin(rad1 + rad2);

    endXVal.textContent = x2.toFixed(1) + ' cm';
    endYVal.textContent = y2.toFixed(1) + ' cm';

    // Jacobian Matrix J = [ [-l1*s1 - l2*s12, -l2*s12], [l1*c1 + l2*c12, l2*c12] ]
    const s1 = Math.sin(rad1), c1 = Math.cos(rad1);
    const s12 = Math.sin(rad1 + rad2), c12 = Math.cos(rad1 + rad2);

    const J11 = (-l1 * s1 - l2 * s12).toFixed(1);
    const J12 = (-l2 * s12).toFixed(1);
    const J21 = (l1 * c1 + l2 * c12).toFixed(1);
    const J22 = (l2 * c12).toFixed(1);

    jacobianVal.innerHTML = `J(θ) = [ [ ${J11}, ${J12} ],<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ ${J21}, ${J22} ] ]`;

    // Check obstacle collision (Safety Envelope in Physical AI)
    const inObstacle = (x2 >= obstacle.x && x2 <= obstacle.x + obstacle.w &&
                        y2 >= obstacle.y && y2 <= obstacle.y + obstacle.h);

    if (inObstacle) {
      safetyStatus.innerHTML = '<span style="color:#fb7185; font-weight:700;">⚠️ COLLISION DETECTED: Safety E-Stop Active</span>';
    } else {
      safetyStatus.innerHTML = '<span style="color:#34d399; font-weight:600;">✓ Normal Operation: Safe Kinematic Workspace</span>';
    }

    // Canvas drawing
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const originX = w / 2;
    const originY = h - 60;

    // Transform math coords (y up) to canvas (y down)
    function toCX(x) { return originX + x; }
    function toCY(y) { return originY - y; }

    // Draw Workspace Reachability Circle (max reach = l1 + l2)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(originX, originY, l1 + l2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Ground / Pedestal Base
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(originX - 45, originY, 90, 20);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.strokeRect(originX - 45, originY, 90, 20);

    // Draw Obstacle Zone
    ctx.fillStyle = inObstacle ? 'rgba(251, 113, 133, 0.35)' : 'rgba(251, 191, 36, 0.15)';
    ctx.strokeStyle = inObstacle ? '#fb7185' : '#fbbf24';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.fillRect(toCX(obstacle.x), toCY(obstacle.y + obstacle.h), obstacle.w, obstacle.h);
    ctx.strokeRect(toCX(obstacle.x), toCY(obstacle.y + obstacle.h), obstacle.w, obstacle.h);
    ctx.setLineDash([]);
    ctx.fillStyle = '#fbbf24';
    ctx.font = '10px Fira Code';
    ctx.fillText('Obstacle', toCX(obstacle.x) + 10, toCY(obstacle.y + obstacle.h / 2));

    // Link 1 (Base to Joint 1)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 7;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(toCX(x1), toCY(y1));
    ctx.stroke();

    // Link 2 (Joint 1 to End-Effector)
    ctx.strokeStyle = inObstacle ? '#fb7185' : '#a78bfa';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(toCX(x1), toCY(y1));
    ctx.lineTo(toCX(x2), toCY(y2));
    ctx.stroke();

    // Base Joint Motor
    ctx.fillStyle = '#0284c7';
    ctx.beginPath();
    ctx.arc(originX, originY, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Joint 1 Motor (Elbow)
    ctx.fillStyle = '#8b5cf6';
    ctx.beginPath();
    ctx.arc(toCX(x1), toCY(y1), 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // End-Effector Gripper
    ctx.fillStyle = inObstacle ? '#fb7185' : '#34d399';
    ctx.beginPath();
    ctx.arc(toCX(x2), toCY(y2), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Gripper fingers
    const angleTip = rad1 + rad2;
    const gripL = 12;
    ctx.strokeStyle = inObstacle ? '#fb7185' : '#34d399';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(toCX(x2), toCY(y2));
    ctx.lineTo(toCX(x2) + gripL * Math.cos(angleTip - 0.4), toCY(y2) - gripL * Math.sin(angleTip - 0.4));
    ctx.moveTo(toCX(x2), toCY(y2));
    ctx.lineTo(toCX(x2) + gripL * Math.cos(angleTip + 0.4), toCY(y2) - gripL * Math.sin(angleTip + 0.4));
    ctx.stroke();
  }

  theta1Slider.addEventListener('input', render);
  theta2Slider.addEventListener('input', render);
  render();
}

/* ==========================================================================
   19. KATEX FORMAL MATHEMATICAL TYPESETTING ENGINE
   ========================================================================== */
function initMathRendering() {
  function applyKatex() {
    if (typeof renderMathInElement === 'function') {
      try {
        renderMathInElement(document.body, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false }
          ],
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
          throwOnError: false
        });
      } catch (err) {
        console.warn('KaTeX auto-render error:', err);
      }
    }
  }

  // Initial render
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyKatex);
  } else {
    applyKatex();
  }

  // Also retry once window is fully loaded (in case KaTeX scripts loaded deferred)
  window.addEventListener('load', applyKatex);
}


/* ==========================================================================
   20. SPOTLIGHT SEARCH, LIGHTBOX MODAL, FLOATING DOCK & SIMULATORS
   ========================================================================== */

const spotlightSearchDatabase = [
  { title: "Five Disciplinary Pillars", category: "Part 1", target: "#part1", desc: "Computer Science, Mathematics, Neuroscience, Cognitive Psychology, Ethics." },
  { title: "Three Logics of Reasoning", category: "Part 1", target: "#part1", desc: "First Principles (deduction), Deep Learning (induction), Analogy (embeddings)." },
  { title: "Regression Analysis Lab", category: "Part 1", target: "#part1", desc: "Linear MSE optimization and Logistic Sigmoid probability mapping." },
  { title: "Bayes' Theorem & Base-Rate Fallacy", category: "Part 1", target: "#bayesCalculatorLab", desc: "Medical diagnostic certainty under low disease prevalence." },
  { title: "Shannon Information Entropy", category: "Part 1", target: "#part1", desc: "H(X) bits measurement, Cross-Entropy Loss, and KL Divergence." },
  { title: "Modern Perceptron Architecture", category: "Part 2", target: "#part2", desc: "Inputs, weighted synapses, bias threshold, and activation function." },
  { title: "XOR Linear Inseparability", category: "Part 2", target: "#part2", desc: "Minsky & Papert 1969 proof and hidden non-linear representations." },
  { title: "Non-Linear Activation Functions", category: "Part 2", target: "#part2", desc: "GELU, ReLU, Sigmoid, Tanh; vanishing gradients in deep networks." },
  { title: "Multi-Layer Perceptrons (MLPs)", category: "Part 2", target: "#part2-mlp", desc: "Matrix projection z = Wa + b and Cybenko Universal Approximation." },
  { title: "5-Step Learning Loop Stepper", category: "Part 2", target: "#part2-learning-cycle", desc: "Forward Prop, Loss, Backpropagation, Gradient Descent, Epochs." },
  { title: "CNN Vision Processing", category: "Part 3", target: "#part3-cnn", desc: "2D spatial convolution (I * W), local receptive fields, and weight sharing." },
  { title: "Vanilla RNN & Sequential Memory", category: "Part 3", target: "#part3-rnn", desc: "Hidden recurrence state ht = tanh(Whh*h + Wxh*x) and BPTT decay." },
  { title: "LSTM Gated Memory Highways", category: "Part 3", target: "#part3-lstm", desc: "Constant Error Carousel governed by Forget, Input, and Output gates." },
  { title: "Reinforcement Learning & MDP", category: "Part 4", target: "#part4", desc: "States, actions, rewards, transition probabilities, and discount factor gamma." },
  { title: "Bellman Optimality Equations", category: "Part 4", target: "#part4", desc: "Dynamic programming backup for state-value V*(s) and action-value Q*(s, a)." },
  { title: "Interactive Q-Learning Gridworld", category: "Part 4", target: "#part4-qgrid", desc: "Live 4x4 simulation exploring exploitation vs. exploration (epsilon-greedy)." },
  { title: "AlphaGo & RLHF Alignment", category: "Part 4", target: "#part4-alphago", desc: "Policy/value networks (Silver 2016) and 3-stage InstructGPT alignment." },
  { title: "Self-Attention Mechanism", category: "Part 5", target: "#part5-attention", desc: "Query, Key, Value mapping and Scaled Dot-Product Attention Softmax(QK^T / sqrt(dk))V." },
  { title: "Transformer Architecture", category: "Part 5", target: "#part5", desc: "Multi-Head Attention, Feed-Forward sublayers, and Layer Normalization." },
  { title: "Chinchilla Scaling Laws", category: "Part 5", target: "#part5-chinchilla", desc: "Hoffmann et al. (2022) compute-optimal parameter-to-token ratio (N ~ D ~ C^0.50)." },
  { title: "Memory Wall & HBM Latency Lab", category: "Part 5", target: "#part5-memwall", desc: "Memory bandwidth bottleneck during autoregressive token generation." },
  { title: "LoRA Parameter-Efficient Fine-Tuning", category: "Part 5", target: "#part5-lora", desc: "Low-rank weight matrix decomposition W = W0 + (alpha/r)*B*A." },
  { title: "NVIDIA H200 SM Architecture", category: "Part 6", target: "#part6", desc: "Streaming Multiprocessors, 4th-Gen Tensor Cores, FP8 acceleration." },
  { title: "H200 AI Supercluster Hierarchy", category: "Part 6", target: "#part6", desc: "NVLink 900 GB/s to 1.8 TB/s, NVSwitch nodes, and 24k GPU scaling." },
  { title: "Autonomous Agent Architecture", category: "Part 7", target: "#part7", desc: "Perception, working memory, reasoning engine, and external tool actuation." },
  { title: "Model Context Protocol (MCP)", category: "Part 7", target: "#part7-mcp", desc: "Open standard for connecting LLMs to local files, databases, and enterprise APIs." },
  { title: "Compounding Error Decay Calculator", category: "Part 7", target: "#part7-comperr", desc: "End-to-end task reliability degradation P = p^n across multi-step execution." },
  { title: "Explore-Plan-Implement-Verify Workflow", category: "Part 7", target: "#part7-coding", desc: "Production coding agent methodology using Claude Code, Cursor, and Antigravity." },
  { title: "Production Prompt Pattern Library", category: "Part 7", target: "#part7-prompts", desc: "Copy-paste production system prompts for senior engineering tasks." },
  { title: "Coding Agents & Developer Tools", category: "Part 8", target: "#part8", desc: "Autonomous coding loops, tool primitives, and background software agents." },
  { title: "Google Gemini & Antigravity Tri-Surface Engine", category: "Part 8", target: "#part8-gemini", desc: "1M+ context window, native tool calling, and Editor-Terminal-Browser execution." },
  { title: "Disciplined Workflow Simulator (Lab 13)", category: "Part 8", target: "#lab-coding-workflow", desc: "Interactive 4-stage engineering lifecycle: Explore, Plan, Implement, Verify." },
  { title: "Coding Agents Comparison Matrix (Lab 14)", category: "Part 8", target: "#lab-tools-matrix", desc: "Compare Gemini, Claude Code, Cursor, Copilot, and Codex side-by-side." },
  { title: "Production Prompt Pattern Library", category: "Part 8", target: "#part8-prompts", desc: "One-click copyable prompts for architecture discovery, TDD, and surgical bug fixing." },
  { title: "Physical AI & Embodiment", category: "Part 9", target: "#part9", desc: "Sensory grounding, actuator dynamics, and material world constraints." },
  { title: "Two-Link Kinematics & Safety Lab", category: "Part 9", target: "#part9", desc: "Forward kinematics equations, Jacobian matrix, and workspace reachability." },
  { title: "PID Closed-Loop Feedback Controller", category: "Part 9", target: "#part9", desc: "Proportional, Integral, and Derivative control simulator for robotic joints." },
  { title: "Vision-Language-Action (VLA) Models", category: "Part 9", target: "#part9", desc: "RT-2, OpenVLA, and multimodal policies predicting robot action tokens." },
  { title: "Comprehensive Academic Glossary", category: "Glossary", target: "#glossarySection", desc: "Searchable dictionary of technical terms across all 9 curriculum parts." },
  { title: "Academic Reference Archive", category: "Part 10", target: "#referencesSection", desc: "32 landmark papers and literature from Shannon (1948) to modern physical AI." }
];


function initSpotlightSearch() {
  const modal = document.getElementById('spotlightModal');
  const input = document.getElementById('spotlightSearchInput');
  const results = document.getElementById('spotlightResultsContainer');
  const trigger = document.getElementById('spotlightTriggerBtn');
  if (!modal || !input) return;

  function openSpotlight() {
    modal.classList.add('active');
    input.value = '';
    renderResults(spotlightSearchDatabase.slice(0, 10));
    setTimeout(() => input.focus(), 60);
  }

  function closeSpotlight() {
    modal.classList.remove('active');
  }

  if (trigger) trigger.addEventListener('click', openSpotlight);

  modal.addEventListener('click', (e) => {
    if (e.target.id === 'spotlightModal') closeSpotlight();
  });

  const kbdClose = modal.querySelector('.spotlight-kbd');
  if (kbdClose) kbdClose.addEventListener('click', closeSpotlight);

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (modal.classList.contains('active')) closeSpotlight();
      else openSpotlight();
    } else if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeSpotlight();
    }
  });

  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q) {
      renderResults(spotlightSearchDatabase.slice(0, 10));
      return;
    }
    const matches = spotlightSearchDatabase.filter(item => 
      item.title.toLowerCase().includes(q) ||
      item.desc.toLowerCase().includes(q) ||
      item.category.toLowerCase().includes(q)
    );
    renderResults(matches.slice(0, 12));
  });

  function renderResults(items) {
    if (!results) return;
    if (items.length === 0) {
      results.innerHTML = '<div style="padding:1.5rem; text-align:center; color:var(--text-muted); font-size:0.9rem;">No matching curriculum modules, formulas, or figures found.</div>';
      return;
    }
    results.innerHTML = items.map(item => `
      <div class="spotlight-item" onclick="jumpToSpotlightTarget('${item.target}')">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600; color:#fff; font-size:0.95rem;">${item.title}</span>
          <span class="spotlight-cat-badge">${item.category}</span>
        </div>
        <p style="font-size:0.82rem; color:#94a3b8; margin:0.3rem 0 0 0; line-height:1.4;">${item.desc}</p>
      </div>
    `).join('');
  }

  window.jumpToSpotlightTarget = function(target) {
    closeSpotlight();
    const el = document.querySelector(target);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      el.classList.add('highlight-flash');
      setTimeout(() => el.classList.remove('highlight-flash'), 2500);
    }
  };
}

function initLightbox() {
  const modal = document.getElementById('lightboxModal');
  const img = document.getElementById('lightboxImg');
  const cap = document.getElementById('lightboxCaption');
  const closeBtn = document.getElementById('lightboxCloseBtn');
  if (!modal || !img) return;

  function openModal(src, captionHtml) {
    img.src = src;
    img.alt = captionHtml ? captionHtml.replace(/<[^>]*>/g, '') : 'Diagram';
    if (cap) cap.innerHTML = captionHtml || '';
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
  }

  document.querySelectorAll('.figure-container').forEach(fig => {
    fig.style.cursor = 'zoom-in';
    fig.addEventListener('click', (e) => {
      if (e.target.tagName === 'A') return;
      const figImg = fig.querySelector('img');
      const figCap = fig.querySelector('.figure-caption');
      if (figImg) {
        openModal(figImg.src, figCap ? figCap.innerHTML : '');
      }
    });
  });

  window.openLightbox = function(src, caption) {
    openModal(src, caption);
  };

  window.closeLightbox = closeModal;

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => {
    if (e.target.id === 'lightboxModal' || e.target.classList.contains('lightbox-content') || e.target.id === 'lightboxCloseBtn') {
      closeModal();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });
}


window.copyPrompt = function(btn) {
  const card = btn.closest('.prompt-card') || btn.parentElement;
  const code = card ? card.querySelector('.prompt-code') : null;
  if (!code) return;
  const text = code.innerText.trim();

  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.innerText;
    btn.innerText = '✓ Copied!';
    btn.style.borderColor = '#34d399';
    btn.style.color = '#34d399';
    setTimeout(() => {
      btn.innerText = orig;
      btn.style.borderColor = '';
      btn.style.color = '';
    }, 2000);
  }).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    btn.innerText = '✓ Copied!';
    setTimeout(() => { btn.innerText = 'Copy Clean Prompt'; }, 2000);
  });
};

/* Q-Learning Gridworld Simulation (Part 4 Upgrade) */
function initQLearningGridworld() {
  const canvas = document.getElementById('gridworldCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const stepBtn = document.getElementById('qStepBtn');
  const runBtn = document.getElementById('qRunBtn');
  const resetBtn = document.getElementById('qResetBtn');
  const epEl = document.getElementById('qEpisodeVal');
  const rewEl = document.getElementById('qRewardVal');

  const gridSize = 4;
  const cellSize = canvas.width / gridSize;
  const goal = { x: 3, y: 3 };
  const obstacles = [{ x: 1, y: 1 }, { x: 2, y: 1 }, { x: 1, y: 2 }];

  let agent = { x: 0, y: 0 };
  let episode = 1;
  let totalReward = 0;
  let isRunning = false;
  let runInterval = null;

  // Initialize Q-table: state (x, y) -> 4 actions [Up, Right, Down, Left]
  const Q = {};
  for (let x = 0; x < gridSize; x++) {
    for (let y = 0; y < gridSize; y++) {
      Q[`${x},${y}`] = [0, 0, 0, 0];
    }
  }

  function isObstacle(x, y) {
    return obstacles.some(o => o.x === x && o.y === y);
  }

  function step() {
    const stateKey = `${agent.x},${agent.y}`;
    const qValues = Q[stateKey];

    // Epsilon-greedy action selection
    let action;
    const eps = Math.max(0.05, 0.5 - episode * 0.01);
    if (Math.random() < eps) {
      action = Math.floor(Math.random() * 4);
    } else {
      let maxQ = -Infinity;
      let best = 0;
      for (let a = 0; a < 4; a++) {
        if (qValues[a] > maxQ) { maxQ = qValues[a]; best = a; }
      }
      action = best;
    }

    // Dynamics: 0=Up, 1=Right, 2=Down, 3=Left
    let nx = agent.x;
    let ny = agent.y;
    if (action === 0) ny = Math.max(0, ny - 1);
    else if (action === 1) nx = Math.min(gridSize - 1, nx + 1);
    else if (action === 2) ny = Math.min(gridSize - 1, ny + 1);
    else if (action === 3) nx = Math.max(0, nx - 1);

    let reward = -0.1;
    let terminal = false;

    if (nx === goal.x && ny === goal.y) {
      reward = 10.0;
      terminal = true;
    } else if (isObstacle(nx, ny)) {
      reward = -5.0;
      nx = agent.x;
      ny = agent.y;
    }

    // Bellman update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]
    const nextStateKey = `${nx},${ny}`;
    const nextMaxQ = Math.max(...Q[nextStateKey]);
    const alpha = 0.2;
    const gamma = 0.9;
    Q[stateKey][action] += alpha * (reward + gamma * nextMaxQ - Q[stateKey][action]);

    agent.x = nx;
    agent.y = ny;
    totalReward += reward;

    if (terminal) {
      episode++;
      agent = { x: 0, y: 0 };
    }

    if (epEl) epEl.textContent = episode;
    if (rewEl) rewEl.textContent = totalReward.toFixed(1);
    renderGrid();
  }

  function renderGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        const px = x * cellSize;
        const py = y * cellSize;

        if (x === goal.x && y === goal.y) {
          ctx.fillStyle = 'rgba(52, 211, 153, 0.35)';
        } else if (isObstacle(x, y)) {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.35)';
        } else {
          ctx.fillStyle = (x + y) % 2 === 0 ? '#0f172a' : '#131b2e';
        }
        ctx.fillRect(px, py, cellSize, cellSize);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        ctx.strokeRect(px, py, cellSize, cellSize);

        // Draw Q-value indicator
        const maxQ = Math.max(...Q[`${x},${y}`]);
        if (maxQ !== 0 && !isObstacle(x, y) && !(x === goal.x && y === goal.y)) {
          ctx.fillStyle = maxQ > 0 ? '#38bdf8' : '#fb7185';
          ctx.font = '10px monospace';
          ctx.fillText(maxQ.toFixed(1), px + 8, py + cellSize - 8);
        }
      }
    }

    // Draw Goal Flag
    ctx.fillStyle = '#34d399';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText('GOAL', goal.x * cellSize + 15, goal.y * cellSize + cellSize / 2 + 6);

    // Draw Agent
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(agent.x * cellSize + cellSize / 2, agent.y * cellSize + cellSize / 2, cellSize / 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  if (stepBtn) stepBtn.addEventListener('click', step);
  if (runBtn) {
    runBtn.addEventListener('click', () => {
      if (!isRunning) {
        isRunning = true;
        runBtn.textContent = 'Pause Training';
        runInterval = setInterval(step, 80);
      } else {
        isRunning = false;
        runBtn.textContent = 'Auto Train Agent';
        clearInterval(runInterval);
      }
    });
  }
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      isRunning = false;
      if (runInterval) clearInterval(runInterval);
      if (runBtn) runBtn.textContent = 'Auto Train Agent';
      episode = 1;
      totalReward = 0;
      agent = { x: 0, y: 0 };
      for (const k in Q) Q[k] = [0, 0, 0, 0];
      if (epEl) epEl.textContent = '1';
      if (rewEl) rewEl.textContent = '0.0';
      renderGrid();
    });
  }

  renderGrid();
}

/* Memory Wall Lab (Part 5 Upgrade) */
function initMemoryWallLab() {
  const precSelect = document.getElementById('quantPrecisionSelect');
  const bwSlider = document.getElementById('hbmBw');
  if (!bwSlider || !precSelect) return;

  function update() {
    const bytesPerParam = parseInt(precSelect.value, 10) / 8;
    const bw = parseFloat(bwSlider.value);
    const bwVal = document.getElementById('hbmBwVal');
    if (bwVal) bwVal.textContent = `${bw} GB/s`;

    const params = 70; // 70B Model
    const footprintGB = params * bytesPerParam;
    const latencySec = footprintGB / bw;
    const tokensPerSec = 1 / latencySec;

    const memFootprint = document.getElementById('memFootprintVal');
    const singleUserLat = document.getElementById('singleUserLatVal');
    const maxTokens = document.getElementById('maxTokensPerSecVal');

    if (memFootprint) memFootprint.textContent = `${footprintGB.toFixed(1)} GB`;
    if (singleUserLat) singleUserLat.textContent = `${(latencySec * 1000).toFixed(1)} ms/token`;
    if (maxTokens) maxTokens.textContent = `${tokensPerSec.toFixed(1)} tokens/sec`;
  }

  precSelect.addEventListener('change', update);
  bwSlider.addEventListener('input', update);
  update();
}

/* PID Controller Simulator (Part 8 Upgrade) */
function initPIDControllerLab() {
  const canvas = document.getElementById('pidCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const kpSlider = document.getElementById('pidKp');
  const kiSlider = document.getElementById('pidKi');
  const kdSlider = document.getElementById('pidKd');
  const stepBtn = document.getElementById('pidStepBtn');
  if (!kpSlider || !kiSlider || !kdSlider) return;

  function render() {
    const Kp = parseFloat(kpSlider.value);
    const Ki = parseFloat(kiSlider.value);
    const Kd = parseFloat(kdSlider.value);

    const kpVal = document.getElementById('pidKpVal');
    const kiVal = document.getElementById('pidKiVal');
    const kdVal = document.getElementById('pidKdVal');
    if (kpVal) kpVal.textContent = Kp.toFixed(1);
    if (kiVal) kiVal.textContent = Ki.toFixed(2);
    if (kdVal) kdVal.textContent = Kd.toFixed(1);

    const target = 1.0;
    const dt = 0.02;
    const steps = 300;
    const response = [];

    let pos = 0;
    let vel = 0;
    let integral = 0;
    let prevError = target - pos;

    for (let t = 0; t < steps; t++) {
      const error = target - pos;
      integral += error * dt;
      const derivative = (error - prevError) / dt;
      const u = Kp * error + Ki * integral + Kd * derivative;
      prevError = error;

      // System Physics: Mass = 1kg, Damping = 0.5
      const accel = u - 0.5 * vel;
      vel += accel * dt;
      pos += vel * dt;

      response.push(pos);
    }

    // Draw on Canvas
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Target Line (Setpoint)
    const targetY = h - (target / 2.0) * (h - 40) - 20;
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(40, targetY);
    ctx.lineTo(w - 20, targetY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#34d399';
    ctx.font = '10px monospace';
    ctx.fillText('Setpoint (Target)', 45, targetY - 6);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let y = 0; y <= 2.0; y += 0.5) {
      const py = h - (y / 2.0) * (h - 40) - 20;
      ctx.beginPath();
      ctx.moveTo(40, py);
      ctx.lineTo(w - 20, py);
      ctx.stroke();
      ctx.fillStyle = '#64748b';
      ctx.fillText(y.toFixed(1), 15, py + 4);
    }

    // Step Response Curve
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < response.length; i++) {
      const px = 40 + (i / steps) * (w - 60);
      const py = h - (response[i] / 2.0) * (h - 40) - 20;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Calculate overshoot and settling time
    const maxVal = Math.max(...response);
    const overshootPct = Math.max(0, ((maxVal - target) / target) * 100);
    const osEl = document.getElementById('pidOvershootVal');
    if (osEl) osEl.textContent = `${overshootPct.toFixed(1)}%`;
  }

  kpSlider.addEventListener('input', render);
  kiSlider.addEventListener('input', render);
  kdSlider.addEventListener('input', render);
  if (stepBtn) stepBtn.addEventListener('click', render);
  render();
}

/* Compounding Error Lab (Part 7 Upgrade) */
function initCompoundingErrorLab() {
  const accSlider = document.getElementById('singleStepAcc');
  const stepsSlider = document.getElementById('taskSteps');
  if (!accSlider || !stepsSlider) return;

  function update() {
    const p = parseFloat(accSlider.value) / 100;
    const n = parseInt(stepsSlider.value, 10);

    const accVal = document.getElementById('singleStepAccVal');
    const stepsVal = document.getElementById('taskStepsVal');
    if (accVal) accVal.textContent = `${(p * 100).toFixed(1)}%`;
    if (stepsVal) stepsVal.textContent = n;

    const overall = Math.pow(p, n);
    const fail = 1 - overall;
    const req = Math.pow(0.90, 1 / n);

    const overallVal = document.getElementById('overallSuccessVal');
    const failVal = document.getElementById('failureProbVal');
    const reqVal = document.getElementById('requiredAccVal');

    if (overallVal) overallVal.textContent = `${(overall * 100).toFixed(1)}%`;
    if (failVal) failVal.textContent = `${(fail * 100).toFixed(1)}%`;
    if (reqVal) reqVal.textContent = `${(req * 100).toFixed(2)}%`;
  }

  accSlider.addEventListener('input', update);
  stepsSlider.addEventListener('input', update);
  update();
}

// Ensure newly added labs are called if present
document.addEventListener('DOMContentLoaded', () => {
  initQLearningGridworld();
  initMemoryWallLab();
  initPIDControllerLab();
  initCompoundingErrorLab();
});


// ====================================================================
// INTERACTIVE LAB 13: DISCIPLINED CODING AGENT WORKFLOW SIMULATOR
// ====================================================================
const cwPhasesData = [
  {
    num: 1,
    title: "1. Explore (Read-Only Exploration)",
    color: "#38bdf8",
    prompt: "Explore the codebase first. Find where the telemetry WebSocket server and authentication handshake are implemented.\nDo not write or modify any code yet.\nReturn:\n1. Affected files, classes, and exported functions.\n2. Architecture design plan with data flow.\n3. Edge cases and test strategy.",
    action: "Agent invokes list_dir and ripgrep grep_search across server/ and auth/. Discovers server/stream.py, auth/jwt_verifier.py, and models/telemetry.py. Maps active socket connection pools and error codes. Zero files mutated.",
    checklist: "✓ Verify zero file modifications or git dirty state.\n✓ Check completeness of file discovery list against known repository architecture.\n✓ Confirm model grasped existing types and abstractions before approving plan phase."
  },
  {
    num: 2,
    title: "2. Plan (Architectural Specification Artifact)",
    color: "#c084fc",
    prompt: "Formulate an architectural implementation plan. Propose:\n1. Exponential backoff equation with jitter: t_k = min(t_max, t_0 * 2^k) ± δ.\n2. Token refresh hook interface.\n3. Blast-radius assessment and regression risk matrix.",
    action: "Agent generates structured markdown artifact implementation_plan.md containing mermaid state diagrams, schema definitions, and unit test plan. Waits for explicit developer approval in Antigravity review panel.",
    checklist: "✓ Inspect edge case handling (network timeout, cluster failover, token expiration).\n✓ Confirm proposed changes strictly localize blast radius.\n✓ Approve plan before authorizing code modification."
  },
  {
    num: 3,
    title: "3. Implement (Surgical Diffs & Test-Driven Development)",
    color: "#34d399",
    prompt: "Approved plan. Write failing unit test test_reconnect_backoff_jitter() in tests/test_telemetry.py first.\nRun pytest to verify failure.\nOnly then implement minimal localized fix in server/telemetry.py with zero placeholders.",
    action: "Agent invokes write_to_file to create tests/test_telemetry.py. Runs pytest in terminal: test fails red as expected. Applies surgical replace_file_content chunk to server/telemetry.py adding jittered backoff. Reruns tests: 14/14 tests pass green.",
    checklist: "✓ Review unified diff chunk-by-chunk.\n✓ Verify zero comments omitting code blocks (// TODO or placeholders).\n✓ Confirm automated unit tests assert real business logic rather than tautologies."
  },
  {
    num: 4,
    title: "4. Verify (Daemon Launch & Visual Browser Subagent)",
    color: "#fbbf24",
    prompt: "Start dev server in background and verify end-to-end telemetry streaming in browser.\nEnsure 0 console errors and capture verification recording.",
    action: "Agent starts dev server daemon on port 8080. Spawns browser_subagent to navigate to client UI, simulate connection drop, verify automatic reconnection within 1.2s, confirm 0 console errors, and record WebP verification artifact.",
    checklist: "✓ Inspect browser verification recording artifact.\n✓ Confirm zero runtime console warnings or errors.\n✓ Approve git commit and pull request merge."
  }
];

let cwCurrentPhase = 0;

window.renderCwPhase = function() {
  const p = cwPhasesData[cwCurrentPhase];
  const numEl = document.getElementById('cwPhaseNumber');
  const titleEl = document.getElementById('cwPhaseTitle');
  const boxEl = document.getElementById('cwPhaseBox');
  const prevBtn = document.getElementById('cwPrevBtn');
  const nextBtn = document.getElementById('cwNextBtn');

  if (numEl) numEl.innerText = p.num;
  if (titleEl) {
    titleEl.innerText = p.title;
    titleEl.style.color = p.color;
  }
  if (prevBtn) prevBtn.disabled = (cwCurrentPhase === 0);
  if (nextBtn) nextBtn.disabled = (cwCurrentPhase === cwPhasesData.length - 1);

  if (boxEl) {
    boxEl.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:1.2rem;">
        <div>
          <span class="part-box-tag" style="color:${p.color};">OPTIMAL DEVELOPER PROMPT</span>
          <pre class="cw-prompt-box">${p.prompt}</pre>
        </div>
        <div>
          <span class="part-box-tag tag-cyan">AUTONOMOUS AGENT ACTIONS &amp; TOOLS</span>
          <div class="cw-action-box">
            ${p.action}
          </div>
        </div>
        <div>
          <span class="part-box-tag tag-emerald">DEVELOPER QUALITY GATE CHECKLIST</span>
          <pre class="cw-checklist-box">${p.checklist}</pre>
        </div>
      </div>
    `;
  }
};

window.cwChangePhase = function(delta) {
  cwCurrentPhase = Math.max(0, Math.min(cwPhasesData.length - 1, cwCurrentPhase + delta));
  window.renderCwPhase();
};

// ====================================================================
// INTERACTIVE LAB 14: LEADING CODING AGENTS COMPARISON MATRIX
// ====================================================================
const toolsMatrixData = [
  {
    id: "gemini",
    name: "Google Antigravity & Gemini",
    vendor: "Google",
    form: "Agent-First IDE & Cloud Platform",
    models: "Gemini 2.5 Pro / Flash & Gemini 3",
    context: "1,000,000+ Tokens",
    surfaces: "Editor, Shell Daemon, Visual Browser Subagent",
    instructions: ".gemini/rules/ & AGENTS.md",
    strengths: "Tri-surface execution, multi-agent manager, visual browser verification recordings, background process daemons"
  },
  {
    id: "claude",
    name: "Claude Code",
    vendor: "Anthropic",
    form: "Terminal CLI Agent",
    models: "Claude 3.7 Sonnet (Hybrid Reasoning)",
    context: "200,000 Tokens (Prompt Caching)",
    surfaces: "Shell / Terminal, Filesystem, Git Worktrees",
    instructions: "CLAUDE.md & AGENTS.md",
    strengths: "High-speed terminal scriptability, unix pipeline integration, deep reasoning on architectural tradeoffs"
  },
  {
    id: "cursor",
    name: "Cursor",
    vendor: "Anysphere",
    form: "Agentic IDE (VS Code Fork)",
    models: "Multi-vendor (Claude, GPT-4o, Custom)",
    context: "128,000 - 200,000 Tokens",
    surfaces: "Editor (Multi-file Composer), Inline Diffs, Terminal",
    instructions: ".cursorrules & AGENTS.md",
    strengths: "Instant inline multi-file diffing, semantic vector code search index, familiar VS Code ecosystem"
  },
  {
    id: "copilot",
    name: "GitHub Copilot",
    vendor: "GitHub / Microsoft",
    form: "In-Editor Extension & Cloud Agent",
    models: "Multi-vendor (OpenAI, Anthropic)",
    context: "64,000 - 128,000 Tokens",
    surfaces: "Editor Sidebar, Inline Tab Autocomplete",
    instructions: ".github/copilot-instructions.md",
    strengths: "Zero-friction enterprise compliance, frictionless inline code completion, ubiquitous IDE support"
  },
  {
    id: "codex",
    name: "OpenAI Codex",
    vendor: "OpenAI",
    form: "Terminal Agent & Web Platform",
    models: "GPT-4o, o3-mini",
    context: "128,000 - 200,000 Tokens",
    surfaces: "Shell / Terminal, Sandbox Containers",
    instructions: "AGENTS.md",
    strengths: "Open-source CLI harness, sandboxed container evaluation, deep mathematical reasoning"
  }
];

window.switchToolsTab = function(filterId, btn) {
  const container = document.getElementById('toolsMatrixWrap');
  const buttons = document.querySelectorAll('.matrix-tab-btn');
  buttons.forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const filtered = (filterId === 'all') ? toolsMatrixData : toolsMatrixData.filter(t => t.id === filterId);

  if (container) {
    let rowsHtml = filtered.map(t => `
      <tr>
        <td>
          <strong style="color:#f8fafc; font-size:0.95rem;">${t.name}</strong><br>
          <span style="font-size:0.76rem; color:#94a3b8;">${t.vendor}</span>
        </td>
        <td><span style="font-size:0.84rem; color:#cbd5e1;">${t.form}</span></td>
        <td><strong style="color:#38bdf8; font-size:0.84rem;">${t.models}</strong></td>
        <td><span style="font-family:var(--font-mono); font-size:0.82rem; color:#34d399;">${t.context}</span></td>
        <td><span style="font-size:0.82rem; color:#cbd5e1;">${t.surfaces}</span></td>
        <td><code style="font-size:0.78rem; background:rgba(255,255,255,0.06); padding:0.15rem 0.4rem; border-radius:4px;">${t.instructions}</code></td>
        <td style="font-size:0.82rem; color:#94a3b8; line-height:1.45;">${t.strengths}</td>
      </tr>
    `).join('');

    container.innerHTML = `
      <table class="tools-matrix-table">
        <thead>
          <tr>
            <th>Tool &amp; Vendor</th>
            <th>Primary Form Factor</th>
            <th>Foundation Models</th>
            <th>Context Window</th>
            <th>Execution Surfaces</th>
            <th>Instruction File</th>
            <th>Distinctive Strengths</th>
          </tr>
        </thead>
        <tbody>
          ${rowsHtml}
        </tbody>
      </table>
    `;
  }
};

// Auto-initialize Part 8 labs on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  if (typeof window.renderCwPhase === 'function') window.renderCwPhase();
  if (typeof window.switchToolsTab === 'function') {
    const defaultBtn = document.querySelector('.matrix-tab-btn.active');
    window.switchToolsTab('all', defaultBtn);
  }
});
