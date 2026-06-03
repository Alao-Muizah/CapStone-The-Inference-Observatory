const API_BASE = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : ''

// ══════════════════════════════════════════════════════════════════════════════
// 1. CONSTANTS AND CONFIGURATION
// ══════════════════════════════════════════════════════════════════════════════

const GROUPS = [
  {
    id: 'deterministic',
    label: 'Group 1 — Deterministic',
    strategies: ['greedy', 'beam_search']
  },
  {
    id: 'sampling',
    label: 'Group 2 — Pure Sampling',
    strategies: ['sampling', 'temperature']
  },
  {
    id: 'basic-constrained',
    label: 'Group 3A — Basic Constrained',
    strategies: ['top_k', 'top_p']
  },
  {
    id: 'hybrid-constrained',
    label: 'Group 3B — Hybrid Constrained',
    strategies: ['top_k_top_p', 'temp_top_k', 'temp_top_p', 'temp_top_k_top_p']
  }
]

const STRATEGY_META = {
  greedy:           { label: 'Greedy',               group: 'deterministic' },
  beam_search:      { label: 'Beam Search',           group: 'deterministic' },
  sampling:         { label: 'Random Sampling',       group: 'sampling' },
  temperature:      { label: 'Temperature',           group: 'sampling' },
  top_k:            { label: 'Top-k',                 group: 'basic-constrained' },
  top_p:            { label: 'Top-p',                 group: 'basic-constrained' },
  top_k_top_p:      { label: 'Top-k + Top-p',         group: 'hybrid-constrained' },
  temp_top_k:       { label: 'Temp + Top-k',          group: 'hybrid-constrained' },
  temp_top_p:       { label: 'Temp + Top-p',          group: 'hybrid-constrained' },
  temp_top_k_top_p: { label: 'Temp + Top-k + Top-p', group: 'hybrid-constrained' },
}

const allOutputs = {}
let analysisReady = false

// ══════════════════════════════════════════════════════════════════════════════
// 2. BACKEND HEALTH CHECK
// ══════════════════════════════════════════════════════════════════════════════

async function checkBackendHealth() {
  try {
    await fetch(`${API_BASE}/health`)
  } catch {
    console.warn('Backend not reachable.')
  }
}

checkBackendHealth()

// ══════════════════════════════════════════════════════════════════════════════
// 3. LIVE TOKEN COUNTER
// ══════════════════════════════════════════════════════════════════════════════

let encoder = null

try {
  encoder = tiktoken.get_encoding('cl100k_base')
} catch (error) {
  console.warn('Tiktoken failed to load, falling back to approximation:', error)
}

function countTokens(text) {
  if (!text || text.trim() === '') return 0
  if (encoder) return encoder.encode(text).length
  return Math.max(1, Math.ceil(text.length / 4))
}

const promptInput = document.getElementById('prompt-input')
const tokenCountDisplay = document.getElementById('token-count')

promptInput.addEventListener('input', function(event) {
  const text = event.target.value
  tokenCountDisplay.textContent = countTokens(text)
})

// ══════════════════════════════════════════════════════════════════════════════
// 4. COLLECT PARAMS AND VALIDATE
// ══════════════════════════════════════════════════════════════════════════════

const runAllBtn = document.getElementById('run-all-btn')
const runAnalysisBtn = document.getElementById('run-analysis-btn')

function collectParams() {
  return {
    prompt:       promptInput.value.trim(),
    max_tokens:   parseInt(document.getElementById('max-tokens').value, 10),
    beam_size:    parseInt(document.getElementById('param-beam-size').value, 10),
    top_k:        parseInt(document.getElementById('param-top-k').value, 10),
    top_p:        parseFloat(document.getElementById('param-top-p').value),
    temperature:  parseFloat(document.getElementById('param-temperature').value),
    tktp_k:       parseInt(document.getElementById('param-tktp-k').value, 10),
    tktp_p:       parseFloat(document.getElementById('param-tktp-p').value),
    ttk_temp:     parseFloat(document.getElementById('param-ttk-temp').value),
    ttk_k:        parseInt(document.getElementById('param-ttk-k').value, 10),
    ttp_temp:     parseFloat(document.getElementById('param-ttp-temp').value),
    ttp_p:        parseFloat(document.getElementById('param-ttp-p').value),
    ttkp_temp:    parseFloat(document.getElementById('param-ttkp-temp').value),
    ttkp_k:       parseInt(document.getElementById('param-ttkp-k').value, 10),
    ttkp_p:       parseFloat(document.getElementById('param-ttkp-p').value),
  }
}

function validateParams(params) {
  if (!params.prompt) {
    alert('Please enter a prompt before running.')
    return false
  }
  if (isNaN(params.max_tokens) || params.max_tokens < 1) {
    alert('Max tokens must be a valid number greater than 0.')
    return false
  }
  if (isNaN(params.top_p) || params.top_p < 0 || params.top_p > 1) {
    alert('Top-p must be between 0 and 1.')
    return false
  }
  if (isNaN(params.temperature) || params.temperature < 0) {
    alert('Temperature must be a valid positive number.')
    return false
  }
  return true
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. CARD STATE MANAGEMENT
// ══════════════════════════════════════════════════════════════════════════════

function setCardState(strategy, state) {
  /*
    Sets the visual state of a strategy card.
    The streaming display inside the card also shows the current state
    until actual tokens arrive.
  */
  const card = document.querySelector(`[data-strategy="${strategy}"]`)
  if (!card) return

  card.classList.remove('pending', 'running', 'done', 'error')
  card.classList.add(state)

  const streamDisplay = document.getElementById(`stream-${strategy}`)
  if (!streamDisplay) return

  if (state === 'pending') {
    streamDisplay.innerHTML = `
      <span style="color:var(--text-muted);font-family:var(--font-mono);font-size:0.82rem;">
        Pending...
      </span>`
  }

  if (state === 'running') {
    streamDisplay.innerHTML = `
      <span class="spinner"></span>
      <span style="color:var(--text-accent);font-family:var(--font-mono);font-size:0.82rem;">
        Running...
      </span>`
  }

  if (state === 'error') {
    streamDisplay.innerHTML = `
      <span style="color:var(--red);font-family:var(--font-mono);font-size:0.82rem;">
        Failed
      </span>`
  }
}

function setAllCardsPending() {
  Object.keys(STRATEGY_META).forEach(function(strategy) {
    setCardState(strategy, 'pending')

    const card = document.querySelector(`[data-strategy="${strategy}"]`)
    if (!card) return

    card.querySelector('.col-token-count').textContent = '—'
    card.querySelector('.col-latency').textContent = '—'
    card.querySelector('.col-toks').textContent = '—'
    card.querySelector('.col-finish').textContent = '—'

    const groupOutput = document.getElementById(`output-text-${strategy}`)
    if (groupOutput) {
      groupOutput.textContent = 'Waiting for results...'
      groupOutput.style.color = ''
    }
  })
}

// ══════════════════════════════════════════════════════════════════════════════
// 6. RUN ALL — MAIN ORCHESTRATOR
// ══════════════════════════════════════════════════════════════════════════════

async function runAll() {
  const params = collectParams()
  if (!validateParams(params)) return

  runAllBtn.disabled = true
  runAllBtn.textContent = '⏳ Running...'
  runAnalysisBtn.disabled = true
  analysisReady = false

  setAllCardsPending()
  clearGroupPanels()

  try {
    for (const group of GROUPS) {
      await runGroup(group, params)
    }

    updateTokenPerSec(Object.values(allOutputs))
    analysisReady = true
    runAnalysisBtn.disabled = false
    runAnalysisBtn.textContent = '🔬 Run Analysis'

  } catch (error) {
    console.error('Run All failed:', error)
  }

  runAllBtn.disabled = false
  runAllBtn.textContent = '⚡ Run All Strategies'
}

async function runGroup(group, params) {
  
  for (const strategy of group.strategies) {

    await runStrategy(strategy, params)

    const output = allOutputs[strategy]
    if (output && !output.error && output.tokens) {
      await streamTokensForStrategy(strategy, output.tokens)
    }
  }

  group.strategies.forEach(function(strategy) {
    const output = allOutputs[strategy]
    if (!output || output.error) return

    const groupOutputText = document.getElementById(`output-text-${strategy}`)
    if (groupOutputText) {
      groupOutputText.textContent = output.text
      groupOutputText.style.color = ''
    }
  })

  const groupOutputs = group.strategies
    .map(function(s) { return allOutputs[s] })
    .filter(function(o) { return o && !o.error })

  if (groupOutputs.length >= 2) {
    await computeWithinGroupSimilarity(group.id, groupOutputs)
  }
}

async function runStrategy(strategy, params) {
  setCardState(strategy, 'running')

  try {
    const response = await fetch(`${API_BASE}/api/run-strategy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strategy, params }),
    })

    if (!response.ok) {
      const err = await response.json()
      throw new Error(err.detail || 'Strategy failed')
    }

    const output = await response.json()
    allOutputs[strategy] = output

    populateCard(strategy, output)
    
  } catch (error) {
    console.error(`Strategy ${strategy} failed:`, error)
    allOutputs[strategy] = { strategy, error: error.message }
    setCardState(strategy, 'error')

    const groupOutput = document.getElementById(`output-text-${strategy}`)
    if (groupOutput) {
      groupOutput.textContent = `Error: ${error.message}`
      groupOutput.style.color = 'var(--red)'
    }
  }
}

runAllBtn.addEventListener('click', runAll)

// ══════════════════════════════════════════════════════════════════════════════
// 7. POPULATE CARD
// ══════════════════════════════════════════════════════════════════════════════

function populateCard(strategy, output) {
  const card = document.querySelector(`[data-strategy="${strategy}"]`)
  if (!card) return

  if (output.error) {
    const groupOutput = document.getElementById(`output-text-${strategy}`)
    if (groupOutput) {
      groupOutput.textContent = `Error: ${output.error}`
      groupOutput.style.color = 'var(--red)'
    }
    return
  }

  card.querySelector('.col-token-count').textContent = output.token_count
  card.querySelector('.col-latency').textContent = `${Math.round(output.latency_ms)}ms`
  card.querySelector('.col-finish').textContent = output.finish_reason

  const tokPerSec = (output.token_count / output.latency_ms) * 1000
  card.querySelector('.col-toks').textContent = `${tokPerSec.toFixed(1)}`
}

// ══════════════════════════════════════════════════════════════════════════════
// 8. STREAMING VISUALIZER
// ══════════════════════════════════════════════════════════════════════════════

function streamTokensForStrategy(strategy, tokens) {
  
  return new Promise(function(resolve) {
    const display = document.getElementById(`stream-${strategy}`)
    if (!display) {
      setCardState(strategy, 'done')
      resolve()
      return
    }

    display.innerHTML = ''
    const DELAY_PER_TOKEN = 60

    if (tokens.length === 0) {
      setCardState(strategy, 'done')
      resolve()
      return
    }

    tokens.forEach(function(token, index) {
      setTimeout(function() {
        const span = document.createElement('span')
        span.textContent = token + ' '
        span.classList.add('token-new')
        display.appendChild(span)
        display.scrollTop = display.scrollHeight

        if (index === tokens.length - 1) {
          setTimeout(function() {
            setCardState(strategy, 'done')
            resolve()
          }, 200)
        }
      }, index * DELAY_PER_TOKEN)
    })
  })
}

// ══════════════════════════════════════════════════════════════════════════════
// 9. CLEAR GROUP PANELS
// ══════════════════════════════════════════════════════════════════════════════

function clearGroupPanels() {
  GROUPS.forEach(function(group) {
    group.strategies.forEach(function(strategy) {
      const streamDisplay = document.getElementById(`stream-${strategy}`)
      if (streamDisplay) streamDisplay.innerHTML = ''
    })

    const simGrid = document.getElementById(`similarity-grid-${group.id}`)
    if (simGrid) simGrid.innerHTML = '<p class="label">Waiting for results...</p>'

    const simAvg = document.getElementById(`similarity-avg-${group.id}`)
    if (simAvg) simAvg.innerHTML = ''
  })

  document.getElementById('attention-display').innerHTML =
    '<p class="label">Waiting for Run Analysis...</p>'
  document.getElementById('consistency-display').innerHTML =
    '<p class="label">Select a strategy and click Run Consistency.</p>'
  document.getElementById('dashboard-table-wrap').innerHTML =
    '<p class="label">Waiting for Run Analysis...</p>'
  document.getElementById('similarity-cross').innerHTML =
    '<p class="label">Waiting for Run Analysis...</p>'
  document.getElementById('similarity-avg-cross').innerHTML = ''
}

// ══════════════════════════════════════════════════════════════════════════════
// 10. WITHIN-GROUP SIMILARITY
// ══════════════════════════════════════════════════════════════════════════════

async function computeWithinGroupSimilarity(groupId, outputs) {
  const simGrid = document.getElementById(`similarity-grid-${groupId}`)
  const simAvg = document.getElementById(`similarity-avg-${groupId}`)
  if (!simGrid) return

  simGrid.innerHTML = '<p class="label">Computing similarity...</p>'

  try {
    const texts = {}
    outputs.forEach(function(output) {
      texts[output.strategy] = output.text
    })

    const response = await fetch(`${API_BASE}/api/similarity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts }),
    })

    if (!response.ok) throw new Error('Similarity request failed')

    const data = await response.json()
    renderSimilarityCards(simGrid, simAvg, data.scores)

  } catch (error) {
    simGrid.innerHTML = `<p class="error-text">Similarity failed: ${error.message}</p>`
  }
}

function renderSimilarityCards(gridEl, avgEl, scores) {
  if (!scores || scores.length === 0) {
    gridEl.innerHTML = '<p class="label">No similarity data.</p>'
    return
  }

  let html = ''
  let total = 0

  scores.forEach(function(item) {
    const labelA = STRATEGY_META[item.strategy_a]
      ? STRATEGY_META[item.strategy_a].label : item.strategy_a
    const labelB = STRATEGY_META[item.strategy_b]
      ? STRATEGY_META[item.strategy_b].label : item.strategy_b

    const percentage = (item.score * 100).toFixed(1)
    total += item.score

    const scoreColor = item.score > 0.8
      ? 'var(--green)'
      : item.score > 0.5
        ? 'var(--text-accent)'
        : 'var(--red)'

    html += `
      <div class="similarity-card">
        <p class="similarity-pair">${labelA}<br>vs<br>${labelB}</p>
        <p class="similarity-score-value" style="color:${scoreColor}">
          ${item.score.toFixed(2)}
        </p>
        <div class="similarity-bar-track">
          <div class="similarity-bar-fill" style="width:${percentage}%"></div>
        </div>
      </div>`
  })

  gridEl.innerHTML = html

  if (avgEl) {
    const avg = (total / scores.length).toFixed(2)
    avgEl.innerHTML = `
      <span class="label">Avg similarity:</span>
      <span class="data" style="margin-left:8px;">${avg}</span>`
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 11. UPDATE AVG TOK/S
// ══════════════════════════════════════════════════════════════════════════════

function updateTokenPerSec(outputs) {
  const display = document.getElementById('tokens-per-sec')
  const validOutputs = outputs.filter(function(o) { return !o.error })

  if (validOutputs.length === 0) {
    display.textContent = '—'
    return
  }

  const speeds = validOutputs.map(function(o) {
    return (o.token_count / o.latency_ms) * 1000
  })

  const avg = speeds.reduce(function(sum, s) {
    return sum + s
  }, 0) / speeds.length

  display.textContent = `${avg.toFixed(1)} tok/s`
}

// ══════════════════════════════════════════════════════════════════════════════
// 12. TRADEOFF DASHBOARD
// ══════════════════════════════════════════════════════════════════════════════

function populateDashboard(outputs) {
  const wrap = document.getElementById('dashboard-table-wrap')

  if (!outputs || outputs.length === 0) {
    wrap.innerHTML = '<p class="label">No data yet.</p>'
    return
  }

  let html = `
    <table class="dashboard-table">
      <thead>
        <tr>
          <th>Strategy</th>
          <th>Group</th>
          <th>Latency</th>
          <th>Tokens</th>
          <th>Tok/s</th>
          <th>Finish</th>
        </tr>
      </thead>
      <tbody>`

  outputs.forEach(function(output) {
    if (output.error) return

    const meta = STRATEGY_META[output.strategy]
    const label = meta ? meta.label : output.strategy
    const group = meta ? meta.group : '—'
    const tokPerSec = ((output.token_count / output.latency_ms) * 1000).toFixed(1)

    html += `
      <tr>
        <td>${label}</td>
        <td>${group}</td>
        <td>${Math.round(output.latency_ms)}ms</td>
        <td>${output.token_count}</td>
        <td>${tokPerSec}</td>
        <td>${output.finish_reason}</td>
      </tr>`
  })

  html += `</tbody></table>`
  wrap.innerHTML = html
}

// ══════════════════════════════════════════════════════════════════════════════
// 13. CONSISTENCY TRACKER — STANDALONE WITH ITS OWN RUN BUTTON
// ══════════════════════════════════════════════════════════════════════════════

async function runConsistency() {
  
  const strategy = document.getElementById('consistency-strategy-select').value
  const params = collectParams()

  if (!params.prompt) {
    alert('Please enter a prompt first.')
    return
  }

  const display = document.getElementById('consistency-display')
  const consistencyBtn = document.getElementById('consistency-run-btn')

  display.innerHTML = '<p class="label">Running 5 rounds...</p>'
  consistencyBtn.disabled = true
  consistencyBtn.textContent = '⏳ Running...'

  try {
    
    const runs = await Promise.all(
      Array(5).fill(0).map(async function(_, i) {
        const response = await fetch(`${API_BASE}/api/run-strategy`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strategy, params }),
        })

        if (!response.ok) throw new Error(`Run ${i + 1} failed`)
        const output = await response.json()
        return { run_number: i + 1, output }
      })
    )

    const meta = STRATEGY_META[strategy]
    const label = meta ? meta.label : strategy

    const texts = runs.map(function(r) {
      return r.output.error ? '' : r.output.text
    })
    const allIdentical = texts.every(function(t) { return t === texts[0] })

    let html = `
      <div style="margin-bottom:12px;">
        <p class="label">${label} — 5 runs</p>
        ${allIdentical
          ? '<p class="secondary" style="margin-top:6px;">All outputs identical — deterministic strategy.</p>'
          : '<p class="secondary" style="margin-top:6px;">Outputs vary across runs — sampling strategy.</p>'
        }
      </div>`

    runs.forEach(function(run) {
      const text = run.output.error
        ? `Error: ${run.output.error}`
        : run.output.text

      html += `
        <div class="consistency-run-item">
          <span class="run-number">#${run.run_number}</span>
          <span class="run-text">${text}</span>
        </div>`
    })

    display.innerHTML = html

  } catch (error) {
    display.innerHTML = `<p class="error-text">Consistency failed: ${error.message}</p>`
    console.error('Consistency error:', error)
  }

  consistencyBtn.disabled = false
  consistencyBtn.textContent = 'Run 5×'
}

document.getElementById('consistency-run-btn')
  .addEventListener('click', runConsistency)

document.getElementById('consistency-strategy-select')
  .addEventListener('change', function() {
    document.getElementById('consistency-display').innerHTML =
      '<p class="label">Click Run 5× to test this strategy.</p>'
  })

// ══════════════════════════════════════════════════════════════════════════════
// 14. ATTENTION HEATMAP
// ══════════════════════════════════════════════════════════════════════════════

async function extractAndRenderAttention(prompt) {
  const attentionDisplay = document.getElementById('attention-display')
  attentionDisplay.innerHTML = '<p class="label">Computing attention matrix...</p>'

  try {
    const response = await fetch(`${API_BASE}/api/attention`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word: '', prompt }),
    })

    if (!response.ok) throw new Error('Attention request failed')

    const data = await response.json()
    renderAttentionHeatmap(data)

  } catch (error) {
    attentionDisplay.innerHTML =
      `<p class="error-text">Attention failed: ${error.message}</p>`
  }
}

function renderAttentionHeatmap(data) {
  const display = document.getElementById('attention-display')

  if (!data || !data.attention_matrix) {
    display.innerHTML = '<p class="label">No attention data available.</p>'
    return
  }

  const tokens = data.tokens

  function buildTable(matrix, title, subtitle) {
    let html = `
      <div class="heatmap-block">
        <p class="heatmap-title">${title}</p>
        <p class="heatmap-subtitle">${subtitle}</p>
        <div style="overflow-x:auto;">
        <table class="attention-table">
          <thead><tr><th></th>`

    tokens.forEach(function(token) {
      html += `<th class="attention-col-header"><span>${token}</span></th>`
    })

    html += `</tr></thead><tbody>`

    matrix.forEach(function(row, rowIndex) {
      html += `<tr>`
      html += `<td class="attention-row-header">${tokens[rowIndex]}</td>`

      const rowMax = Math.max(...row)

      row.forEach(function(weight) {
        const normalized = rowMax > 0 ? weight / rowMax : 0
        const intensity = Math.min(5, Math.round(normalized * 5))
        html += `
          <td class="attention-cell attn-${intensity}"
              title="${weight.toFixed(4)}">
          </td>`
      })

      html += `</tr>`
    })

    html += `</tbody></table></div></div>`
    return html
  }

  let html = `
    <p class="secondary" style="margin-bottom:16px;font-size:0.8rem;">
      Model: Qwen2.5-0.5B — ${data.num_layers} layers × ${data.num_heads} heads
      = ${data.num_layers * data.num_heads} attention matrices
    </p>
    <div class="heatmap-grid">`

  html += buildTable(
    data.layer0_head0,
    'Layer 0 — Head 0',
    'Raw attention weights for the first layer and head'
  )

  html += buildTable(
    data.attention_matrix,
    'Averaged — All Layers & Heads',
    'Normalized average across all ' + (data.num_layers * data.num_heads) + ' matrices'
  )

  html += `</div>`
  display.innerHTML = html
}

// ══════════════════════════════════════════════════════════════════════════════
// 15. CROSS-GROUP SIMILARITY
// ══════════════════════════════════════════════════════════════════════════════

async function computeCrossGroupSimilarity() {
  const representatives = ['greedy', 'temperature', 'top_k', 'temp_top_k_top_p']
  const gridEl = document.getElementById('similarity-cross')
  const avgEl = document.getElementById('similarity-avg-cross')

  gridEl.innerHTML = '<p class="label">Computing cross-group similarity...</p>'

  try {
    const texts = {}
    representatives.forEach(function(strategy) {
      const output = allOutputs[strategy]
      if (output && !output.error) {
        texts[strategy] = output.text
      }
    })

    if (Object.keys(texts).length < 2) {
      gridEl.innerHTML = '<p class="label">Not enough outputs to compare.</p>'
      return
    }

    const response = await fetch(`${API_BASE}/api/similarity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts }),
    })

    if (!response.ok) throw new Error('Cross-group similarity failed')

    const data = await response.json()
    renderSimilarityCards(gridEl, avgEl, data.scores)

  } catch (error) {
    gridEl.innerHTML =
      `<p class="error-text">Cross-group similarity failed: ${error.message}</p>`
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// 16. RUN ANALYSIS
// ══════════════════════════════════════════════════════════════════════════════

async function runAnalysis() {
  if (!analysisReady) return

  runAnalysisBtn.disabled = true
  runAnalysisBtn.textContent = '⏳ Analysing...'

  await extractAndRenderAttention(promptInput.value.trim())
  populateDashboard(Object.values(allOutputs))
  await computeCrossGroupSimilarity()

  runAnalysisBtn.disabled = false
  runAnalysisBtn.textContent = '🔬 Run Analysis'
}

runAnalysisBtn.addEventListener('click', runAnalysis)