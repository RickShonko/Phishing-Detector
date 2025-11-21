// app.js — Handles UI interactions and calls backend /analyze-message

const analyzeBtn = document.getElementById('analyzeBtn');
const messageInput = document.getElementById('messageInput');
const resultBody = document.getElementById('resultBody');
const scoreBadge = document.getElementById('scoreBadge');

const senderSelect = document.getElementById('senderSelect');
const senderOther = document.getElementById('senderOther');

// Show/hide 'Other' input dynamically
senderSelect.addEventListener('change', () => {
  if (senderSelect.value === 'Other') {
    senderOther.style.display = 'block';
  } else {
    senderOther.style.display = 'none';
    senderOther.value = '';
  }
});

function setBadge(className, text){
  scoreBadge.className = 'badge ' + className;
  scoreBadge.textContent = text;
}

function showPlaceholder(text){
  resultBody.innerHTML = `<div class="placeholder">${text}</div>`;
}

function renderResult(data){
  const cls = data.classification || 'Unknown';
  let cclass = 'neutral';
  if (cls.toLowerCase().includes('phishing') || cls.toLowerCase().includes('high')) cclass = 'red';
  else if (cls.toLowerCase().includes('suspicious') || cls.toLowerCase().includes('medium')) cclass = 'orange';
  else if (cls.toLowerCase().includes('legitimate') || cls.toLowerCase().includes('safe')) cclass = 'green';

  setBadge(cclass, cls);

  const html = `
    <div class="result-block">
      <h3>Explanation</h3>
      <div class="result-text">${escapeHtml(data.explanation || 'No explanation provided.')}</div>
    </div>
    <div class="result-block">
      <h3>Recommended action</h3>
      <div class="result-text">${escapeHtml(data.recommended_action || 'No action provided.')}</div>
    </div>
    <div class="result-block">
      <h3>Risk score</h3>
      <div class="result-text">${(data.risk_score !== undefined) ? (Number(data.risk_score).toFixed(2)) : 'N/A'}</div>
    </div>
  `;

  resultBody.innerHTML = html;
}

function escapeHtml(unsafe){
  return unsafe
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('\"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function analyzeMessage(){
  const message = messageInput.value.trim();
  if (!message){
    showPlaceholder('Please enter a message to analyze.');
    return;
  }

  let sender = senderSelect.value;
  if(sender === 'Other') sender = senderOther.value.trim();

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  setBadge('neutral','Analyzing...');
  showPlaceholder('Contacting backend...');

  try{
    let payload = { 
      message,
      sender
    };

    const res = await fetch('/analyze-message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok){
      const txt = await res.text();
      throw new Error(txt || res.statusText);
    }

    const data = await res.json();
    renderResult(data);

  }catch(err){
    showPlaceholder('Error: ' + (err.message || err));
    setBadge('neutral','Error');
  }finally{
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🛡️ Analyze Message';
  }
}

analyzeBtn.addEventListener('click', analyzeMessage);

// allow Ctrl/Cmd+Enter to submit
messageInput.addEventListener('keydown', (e)=>{
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeMessage();
});

setBadge('neutral','No analysis yet');
showPlaceholder('Paste a message and click Analyze to start.');
