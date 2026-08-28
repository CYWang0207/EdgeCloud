const shell = document.querySelector('.demo-shell');
const timeline = document.getElementById('timeline');
const playButton = document.getElementById('playButton');
const resetButton = document.getElementById('resetButton');
const exportButton = document.getElementById('exportButton');
const playIcon = document.getElementById('playIcon');
const playText = document.getElementById('playText');
const timeDisplay = document.getElementById('timeDisplay');
const eventLog = document.getElementById('eventLog');
const stageHeader = document.querySelector('.stage-header');
const timelineStageButtons = [...document.querySelectorAll('.timeline-stage')];
const causalChain = document.querySelector('.causal-chain');
const causalSteps = [...document.querySelectorAll('.causal-chain li')];
let causalActiveCount = 0;
const menuToggle = document.getElementById('menuToggle');
const menuClose = document.getElementById('menuClose');
const siteMenu = document.getElementById('siteMenu');
const menuScrim = document.getElementById('menuScrim');
const modelnetCasePicker = document.getElementById('modelnetCasePicker');
const modelnetEvidenceNote = document.getElementById('modelnetEvidenceNote');
const cameraImages = [...document.querySelectorAll('.camera-image')];
const frameLabels = [...document.querySelectorAll('.frame-label')];
const driftButtons = [...document.querySelectorAll('.drift-picker [data-drift]')];
const themedSections = [...document.querySelectorAll('[data-page-theme]')];
let themeFrame = 0;

function updatePageTheme() {
  themeFrame = 0;
  const probeY = window.innerHeight * 0.52;
  const activeSection = themedSections.find(section => {
    const rect = section.getBoundingClientRect();
    return rect.top <= probeY && rect.bottom > probeY;
  }) || themedSections[themedSections.length - 1];
  document.body.dataset.pageTheme = activeSection.dataset.pageTheme;
}

function requestThemeUpdate() {
  if (themeFrame) return;
  themeFrame = window.requestAnimationFrame(updatePageTheme);
}

updatePageTheme();
window.addEventListener('scroll', requestThemeUpdate, { passive: true });
window.addEventListener('resize', requestThemeUpdate);

const ui = {
  stageCode: document.getElementById('stageCode'),
  stageTitle: document.getElementById('stageTitle'),
  predictionMetricLabel: document.getElementById('predictionMetricLabel'),
  prediction: document.getElementById('prediction'),
  predictionNote: document.getElementById('predictionNote'),
  confidenceLabel: document.getElementById('confidenceLabel'),
  confidenceText: document.getElementById('confidenceText'),
  confidenceBar: document.getElementById('confidenceBar'),
  viewAction: document.getElementById('viewAction'),
  tokenAction: document.getElementById('tokenAction'),
  cloudAction: document.getElementById('cloudAction'),
  driftMetricLabel: document.getElementById('driftMetricLabel'),
  driftScore: document.getElementById('driftScore'),
  edgeLatency: document.getElementById('edgeLatency'),
  bandwidth: document.getElementById('bandwidth'),
  adapterState: document.getElementById('adapterState'),
};

const driftNames = {
  illumination: '光照漂移',
  motion: '运动模糊',
  noise: '传感器噪声',
};

const driftOptions = {
  boxcars: [
    { key: 'motion', label: '运动模糊' },
    { key: 'illumination', label: '光照' },
    { key: 'noise', label: '传感器噪声' },
  ],
  modelnet: [
    { key: 'illumination', label: '暗光' },
    { key: 'motion', label: '运动模糊' },
    { key: 'noise', label: '传感器噪声' },
  ],
};

const modelnetDrifts = {
  illumination: { severity: 1.0, baseline: 89.141, adapter: 92.301, formal: true },
  motion: { severity: 0.8, baseline: null, adapter: null, formal: false },
  noise: { severity: 0.4, baseline: 84.400, adapter: 92.990, formal: true },
};

const modelnetSample = {
  id: 'airplane_0627',
  label: 'airplane',
  path: 'assets/modelnet40/airplane_0627/clean',
};

const boxcarFrameLabels = [
  'RIGHT FRONT / 右前方',
  'LEFT FRONT / 左前方',
  'LEFT REAR / 左后方',
  'RIGHT REAR / 右后方',
];

const stages = [
  { from: 0, key: 'sense', code: 'S1', title: '边缘接收漂移输入', confidence: 68, prediction: 'Audi', note: '漂移输入进入边缘端', v: '1111', k: '1.00', u: '0', drift: .58, latency: 80, bandwidth: 24, adapter: 'BASELINE', network: 'NETWORK / STABLE' },
  { from: 3, key: 'drift', code: 'S2', title: '检测到场景漂移', confidence: 61, prediction: 'Audi', note: '置信度下降 / 漂移触发', v: '1111', k: '.80', u: '0', drift: .67, latency: 80, bandwidth: 18.6, adapter: 'BASELINE', network: 'NETWORK / JITTER' },
  { from: 5, key: 'schedule', code: 'S3', title: '联合调度决策', confidence: 59, prediction: 'Audi', note: '低价值视图已休眠', v: '1011', k: '.35', u: '1', drift: .72, latency: 73, bandwidth: 16.8, adapter: 'REFRESH QUEUED', network: 'NETWORK / JITTER' },
  { from: 7, key: 'upload', code: 'S4', title: '代表样本上行', confidence: 60, prediction: 'Audi', note: '前台继续本地自治', v: '1011', k: '.35', u: '1', drift: .70, latency: 73, bandwidth: 11.2, adapter: 'UPLOADING', network: 'NETWORK / 5 MB QUERY' },
  { from: 10, key: 'distill', code: 'S5', title: '云端教师蒸馏', confidence: 63, prediction: 'Audi', note: '后台时间轴压缩回放', v: '1011', k: '.35', u: '1', drift: .69, latency: 73, bandwidth: 14.9, adapter: 'TRAINING', network: 'NETWORK / ASYNC' },
  { from: 12, key: 'recovery', code: 'S6', title: 'Adapter 热加载恢复', confidence: 92, prediction: 'Porsche', note: '品牌识别恢复 / 占位', v: '1111', k: '.55', u: '0', drift: .19, latency: 79, bandwidth: 22.4, adapter: 'CLOUD-V2 / 1.2 MB', network: 'NETWORK / STABLE' },
  { from: 15, key: 'done', code: 'OK', title: '闭环演示完成', confidence: 92, prediction: 'Porsche', note: '运行结果可导出', v: '1111', k: '.55', u: '0', drift: .16, latency: 79, bandwidth: 24, adapter: 'CLOUD-V2 / ACTIVE', network: 'NETWORK / STABLE' },
];

const state = {
  time: 0,
  playing: false,
  lastTick: 0,
  stageKey: 'idle',
  scene: 'boxcars',
  drift: 'motion',
  eventSequence: 1,
  events: [{ id: 1, type: 'neutral', time: 0, message: '系统已就绪' }],
};

const eventMessages = {
  sense: '边缘 MV-ViT 开始多视图推理',
  drift: '检测到输入分布漂移',
  schedule: 'Actor-Critic 切换至端云协同',
  upload: '代表性轨迹进入异步上行队列',
  distill: '云端教师开始 Adapter 蒸馏回放',
  recovery: '1.2 MB Adapter 已热加载',
  done: '本次闭环运行完成',
};

const eventTypes = {
  sense: 'info',
  drift: 'warning',
  schedule: 'info',
  upload: 'info',
  distill: 'info',
  recovery: 'success',
  done: 'success',
};

const adapterLabels = {
  BASELINE: '基线模型',
  'REFRESH QUEUED': '更新待下发',
  UPLOADING: '样本上传中',
  TRAINING: '云端蒸馏中',
  'CLOUD-V2 / 1.2 MB': 'CLOUD-V2 · 已下发',
  'CLOUD-V2 / ACTIVE': 'CLOUD-V2 · 已启用',
};

function currentStage(time) {
  let result = stages[0];
  stages.forEach(stage => { if (time >= stage.from) result = stage; });
  return result;
}

function interpolate(start, end, t) {
  return start + (end - start) * Math.max(0, Math.min(1, t));
}

function updateViewSelection(mask) {
  const selectedViews = [];
  ui.viewAction.querySelectorAll('.view-chip').forEach((chip, index) => {
    const active = mask[index] === '1';
    chip.classList.toggle('is-active', active);
    chip.setAttribute('aria-hidden', String(!active));
    if (active) selectedViews.push(`视角${index + 1}`);
  });
  ui.viewAction.setAttribute('aria-label', `已选择${selectedViews.join('、')}`);
}

function configureDriftButtons(scene) {
  const options = driftOptions[scene];
  if (!options.some(option => option.key === state.drift)) state.drift = options[0].key;

  driftButtons.forEach((button, index) => {
    const option = options[index];
    button.dataset.drift = option.key;
    button.textContent = option.label;
    button.classList.toggle('active', state.drift === option.key);
  });
}

function renderCameraScene() {
  cameraImages.forEach((image, index) => {
    if (state.scene === 'modelnet') {
      const path = `${modelnetSample.path}/view-${index + 1}.webp`;
      const value = `url("${path}")`;
      image.style.backgroundImage = value;
      image.style.setProperty('--frame-image', value);
    } else {
      image.style.backgroundImage = '';
      image.style.removeProperty('--frame-image');
    }
  });

  frameLabels.forEach((label, index) => {
    label.textContent = state.scene === 'modelnet'
      ? `VIEW ${String(index + 1).padStart(2, '0')} / 视图${['一', '二', '三', '四'][index]}`
      : boxcarFrameLabels[index];
  });
}

function updateFrameStates(stage) {
  document.querySelectorAll('.frame-state').forEach((item, index) => {
    if (state.scene === 'modelnet') {
      if (['sense', 'drift', 'schedule', 'upload', 'distill'].includes(stage.key)) {
        item.textContent = 'DRIFT';
        item.style.color = '#f2b84b';
      } else if (['recovery', 'done'].includes(stage.key)) {
        item.textContent = modelnetDrifts[state.drift].formal ? 'ADAPTED' : 'VISUAL';
        item.style.color = modelnetDrifts[state.drift].formal ? '#67d391' : '#98a1a8';
      } else {
        item.textContent = 'ACTIVE';
        item.style.color = '#67d391';
      }
      return;
    }

    if (['sense', 'drift', 'schedule', 'upload', 'distill'].includes(stage.key)) {
      item.textContent = index === 1 && ['schedule', 'upload', 'distill'].includes(stage.key) ? 'SLEEP' : 'DRIFT';
      item.style.color = index === 1 ? '#98a1a8' : '#f2b84b';
    } else if (['recovery', 'done'].includes(stage.key)) {
      item.textContent = 'RECOVERED';
      item.style.color = '#67d391';
    } else {
      item.textContent = 'ACTIVE';
      item.style.color = '#67d391';
    }
  });
}

function pushEvent(stage) {
  if (!eventMessages[stage.key] || state.events.some(item => item.key === stage.key)) return;
  state.eventSequence += 1;
  state.events.push({ id: state.eventSequence, key: stage.key, type: eventTypes[stage.key], time: state.time, message: eventMessages[stage.key] });
  state.events = state.events.slice(-40);
}

function renderLog() {
  eventLog.innerHTML = state.events.map((item, index) => `<p class="log-${item.type || 'info'} ${index === state.events.length - 1 ? 'is-new' : ''}"><b class="log-index">${item.id}</b><span>${item.message}</span></p>`).join('');
  requestAnimationFrame(() => eventLog.scrollTo({ top: eventLog.scrollHeight, behavior: 'smooth' }));
}

function animateStageChange() {
  stageHeader.classList.remove('is-changing');
  void stageHeader.offsetWidth;
  stageHeader.classList.add('is-changing');
}

function updateTimelineStage(stageKey) {
  timelineStageButtons.forEach(button => {
    const active = button.dataset.stage === stageKey;
    button.classList.toggle('is-active', active);
    if (active) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
}

function updateCausalLabels() {
  const labels = state.scene === 'modelnet'
    ? ['输入发生漂移', '漂移强度升高', '视角2关闭', 'Token 保留降低', '云端协同开启']
    : ['置信度下降', '漂移分数升高', '视角2关闭', 'Token 保留降低', '云端协同开启'];
  causalSteps.forEach((step, index) => { step.textContent = labels[index]; });
}

function updateCausalFlow(stageKey, stageChanged) {
  const activeCounts = { sense: 0, drift: 2, schedule: 5, upload: 5, distill: 5, recovery: 5, done: 5 };
  const nextCount = activeCounts[stageKey] ?? 0;
  const sweepFirst = stageChanged && stageKey === 'drift';
  const sweepRest = stageChanged && stageKey === 'schedule';

  if (sweepFirst || sweepRest) {
    causalChain.classList.remove('is-sweeping-first', 'is-sweeping-rest');
    void causalChain.offsetWidth;
    causalChain.classList.add(sweepFirst ? 'is-sweeping-first' : 'is-sweeping-rest');
  }

  causalSteps.forEach((step, index) => {
    const active = index < nextCount;
    const inSweep = sweepFirst ? index < 2 : sweepRest ? index >= 1 : index >= causalActiveCount;
    const entering = stageChanged && active && inSweep;
    step.classList.toggle('is-active', active);
    step.classList.toggle('has-active-next', active && index + 1 < nextCount);
    step.classList.remove('is-entering');
    if (entering) {
      const sequenceIndex = sweepRest ? index - 1 : index - causalActiveCount;
      step.style.setProperty('--causal-delay', `${Math.max(0, sequenceIndex) * 230}ms`);
      void step.offsetWidth;
      step.classList.add('is-entering');
    }
  });

  causalActiveCount = nextCount;
}

function update(time, addEvents = true) {
  state.time = Math.max(0, Math.min(15, time));
  const stage = currentStage(state.time);
  const nextIndex = Math.min(stages.indexOf(stage) + 1, stages.length - 1);
  const next = stages[nextIndex];
  const span = Math.max(.01, next.from - stage.from);
  const progress = next === stage ? 1 : (state.time - stage.from) / span;
  const confidence = Math.round(interpolate(stage.confidence, next.confidence, progress));
  const drift = interpolate(stage.drift, next.drift, progress);
  const bandwidth = interpolate(stage.bandwidth, next.bandwidth, progress);

  const stageChanged = shell.dataset.stage !== stage.key;
  shell.dataset.stage = stage.key;
  shell.dataset.drift = state.drift;
  shell.dataset.scene = state.scene;
  ui.stageCode.textContent = stage.code;
  renderCameraScene();

  if (state.scene === 'modelnet') {
    const driftData = modelnetDrifts[state.drift];
    const recovered = ['recovery', 'done'].includes(stage.key);
    const accuracy = recovered ? driftData.adapter : driftData.baseline;

    ui.stageTitle.textContent = `${stage.title} / ModelNet40`;
    ui.predictionMetricLabel.textContent = 'SAMPLE RESULT · 单样本结果';
    ui.prediction.textContent = modelnetSample.label;
    ui.predictionNote.textContent = driftData.formal
      ? (recovered ? 'Adapter 保持正确识别' : '漂移输入 / 基线模型推理')
      : '真实类别 / 运动模糊仅作视觉演示';
    ui.confidenceLabel.textContent = accuracy === null
      ? 'OFFICIAL METRIC · 正式指标'
      : 'OFFICIAL TEST ACCURACY · 全量测试准确率';
    ui.confidenceText.textContent = accuracy === null ? 'N/A' : `${accuracy.toFixed(1)}%`;
    ui.confidenceBar.style.width = accuracy === null ? '0%' : `${accuracy}%`;
    ui.confidenceBar.style.background = accuracy === null ? '#98a1a8' : (recovered ? '#67d391' : '#48c7dc');
    ui.driftMetricLabel.innerHTML = '漂移强度<small>DRIFT SEVERITY</small>';
    ui.driftScore.textContent = recovered ? '0.0' : driftData.severity.toFixed(1);
    ui.adapterState.textContent = recovered
      ? (driftData.formal ? 'MODELNET-V2 · 已启用' : '仅视觉演示')
      : (adapterLabels[stage.adapter] || stage.adapter);
  } else {
    ui.stageTitle.textContent = stage.title;
    ui.predictionMetricLabel.textContent = 'EDGE PREDICTION · 边缘预测';
    ui.prediction.textContent = stage.prediction;
    ui.predictionNote.textContent = stage.note;
    ui.confidenceLabel.textContent = 'CONFIDENCE · 置信度';
    ui.confidenceText.textContent = `${confidence}%`;
    ui.confidenceBar.style.width = `${confidence}%`;
    ui.confidenceBar.style.background = confidence < 70 ? '#f2b84b' : '#67d391';
    ui.driftMetricLabel.innerHTML = '漂移分数<small>E<sub>drift</sub></small>';
    ui.driftScore.textContent = drift.toFixed(2);
    ui.adapterState.textContent = adapterLabels[stage.adapter] || stage.adapter;
  }

  updateViewSelection(stage.v);
  ui.tokenAction.textContent = `${Math.round(Number(stage.k) * 100)}%`;
  ui.cloudAction.textContent = stage.u === '1' ? '端云协同' : '本地自治';
  ui.edgeLatency.textContent = `${stage.latency} ms`;
  ui.bandwidth.textContent = `${bandwidth.toFixed(1)} Mbps`;
  const completion = Math.round((state.time / 15) * 100);
  timeline.value = completion;
  timeline.setAttribute('aria-valuetext', `${completion}%`);
  timeDisplay.textContent = `${completion}%`;
  updateFrameStates(stage);
  updateTimelineStage(stage.key);
  updateCausalFlow(stage.key, stageChanged);
  if (stageChanged) animateStageChange();

  if (addEvents && stage.key !== state.stageKey) {
    state.stageKey = stage.key;
    pushEvent(stage);
    renderLog();
  }
}

function updatePlayButton() {
  const idleLabel = state.time >= 15 ? '重新播放' : state.time > 0 ? '继续' : '开始';
  playIcon.textContent = state.playing ? 'Ⅱ' : '▶';
  playText.textContent = state.playing ? '暂停' : idleLabel;
  playButton.setAttribute('aria-label', state.playing ? '暂停回放' : `${idleLabel}回放`);
}

function setPlaying(value) {
  if (value && state.time >= 15) reset(false);
  state.playing = value;
  updatePlayButton();
  if (value) {
    state.lastTick = performance.now();
    requestAnimationFrame(tick);
  }
}

function tick(now) {
  if (!state.playing) return;
  const delta = (now - state.lastTick) / 1000;
  state.lastTick = now;
  update(state.time + delta);
  if (state.time >= 15) {
    setPlaying(false);
    return;
  }
  requestAnimationFrame(tick);
}

function reset(clearScene = false) {
  state.time = 0;
  setPlaying(false);
  state.stageKey = 'idle';
  state.eventSequence = 1;
  state.events = [{ id: 1, type: 'neutral', time: 0, message: '系统已重置' }];
  if (clearScene) state.scene = 'boxcars';
  renderLog();
  update(0);
}

playButton.addEventListener('click', () => setPlaying(!state.playing));
resetButton.addEventListener('click', () => reset(false));
timeline.addEventListener('input', event => {
  setPlaying(false);
  const selectedTime = (Number(event.target.value) / 100) * 15;
  state.stageKey = currentStage(selectedTime).key;
  update(selectedTime, false);
  updatePlayButton();
});

timelineStageButtons.forEach(button => {
  button.addEventListener('click', () => {
    setPlaying(false);
    const selectedTime = Number(button.dataset.time);
    state.stageKey = currentStage(selectedTime).key;
    update(selectedTime, false);
    updatePlayButton();
  });
});

driftButtons.forEach(button => {
  button.addEventListener('click', () => {
    driftButtons.forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    state.drift = button.dataset.drift;
    shell.dataset.drift = state.drift;
    state.eventSequence += 1;
    state.events.push({ id: state.eventSequence, type: 'info', time: state.time, message: `漂移预设切换：${driftNames[state.drift]}` });
    state.events = state.events.slice(-40);
    renderLog();
    update(state.time, false);
  });
});

document.querySelectorAll('.segmented [data-scene]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.segmented [data-scene]').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    state.scene = button.dataset.scene;
    configureDriftButtons(state.scene);
    shell.dataset.scene = state.scene;
    modelnetCasePicker.hidden = state.scene !== 'modelnet';
    modelnetEvidenceNote.hidden = state.scene !== 'modelnet';
    updateCausalLabels();
    state.eventSequence += 1;
    state.events.push({ id: state.eventSequence, type: 'info', time: state.time, message: state.scene === 'boxcars' ? '切换至 BoxCars116k' : '切换至 ModelNet40 / airplane_0627' });
    state.events = state.events.slice(-40);
    renderLog();
    update(state.time, false);
  });
});

exportButton.addEventListener('click', () => {
  const stage = currentStage(state.time);
  const payload = {
    format_version: 1,
    prototype: state.scene !== 'modelnet' || !modelnetDrifts[state.drift].formal,
    generated_at: new Date().toISOString(),
    scene: state.scene,
    drift: state.drift,
    replay_time_s: Number(state.time.toFixed(1)),
    stage: stage.key,
    sample: state.scene === 'modelnet'
      ? { id: modelnetSample.id, true_label: modelnetSample.label, category: 'clean_control' }
      : null,
    action: { v_t: stage.v, k_t: Number(stage.k), u_t: Number(stage.u) },
    metrics: {
      displayed_metric: state.scene === 'modelnet' ? 'aggregate_test_accuracy_percent' : 'confidence_percent',
      confidence_percent: Number.isNaN(Number(ui.confidenceText.textContent.replace('%', '')))
        ? null
        : Number(ui.confidenceText.textContent.replace('%', '')),
      drift_score: Number(ui.driftScore.textContent),
      edge_latency_ms: stage.latency,
      bandwidth_mbps: Number(ui.bandwidth.textContent.split(' ')[0]),
      adapter_state: ui.adapterState.textContent,
    },
    events: state.events,
    notice: state.scene === 'modelnet'
      ? 'ModelNet40 四视图来自官方测试交付包；暗光与噪声对应正式条件，运动模糊仅作视觉演示；15 秒为展示时间轴。'
      : '当前为概念占位数据，待替换真实 BoxCars 轨迹和实验日志。',
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `edgecloud-demo-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

function openMenu() {
  menuScrim.hidden = false;
  requestAnimationFrame(() => {
    menuScrim.classList.add('is-visible');
    siteMenu.classList.add('is-open');
  });
  document.body.classList.add('menu-open');
  menuToggle.setAttribute('aria-expanded', 'true');
  siteMenu.setAttribute('aria-hidden', 'false');
  siteMenu.inert = false;
  menuClose.focus();
}

function closeMenu() {
  menuScrim.classList.remove('is-visible');
  siteMenu.classList.remove('is-open');
  document.body.classList.remove('menu-open');
  menuToggle.setAttribute('aria-expanded', 'false');
  siteMenu.setAttribute('aria-hidden', 'true');
  siteMenu.inert = true;
  window.setTimeout(() => {
    if (!siteMenu.classList.contains('is-open')) menuScrim.hidden = true;
  }, 320);
}

menuToggle.addEventListener('click', openMenu);
menuClose.addEventListener('click', closeMenu);
menuScrim.addEventListener('click', closeMenu);
siteMenu.querySelectorAll('nav a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && siteMenu.classList.contains('is-open')) {
    closeMenu();
    menuToggle.focus();
  }
});

configureDriftButtons('boxcars');
updateCausalLabels();
renderLog();
update(0);
