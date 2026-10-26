'use strict';

const $ = (id) => document.getElementById(id);
const state = {documents: [], selected: null, status: null, busy: false, deleteId: null};
const formatNumber = (number) => new Intl.NumberFormat().format(number || 0);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function notify(message, error = false) {
  $('notice').textContent = message;
  $('notice').className = error ? 'notice error' : 'notice';
  $('notice').hidden = !message;
}

async function request(path, options = {}) {
  const response = await fetch(path, {credentials: 'same-origin', ...options});
  let payload;
  try { payload = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
  if (!response.ok) {
    const detail = payload.detail;
    const message = Array.isArray(detail) ? detail.map((item) => `${item.loc?.slice(1).join('.') || 'Request'}: ${item.msg}`).join('; ') : detail;
    throw new Error(message || `Request failed (${response.status}).`);
  }
  return payload;
}

const post = (path, data = {}) => request(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});

async function refresh() {
  const [library, status] = await Promise.all([request('/api/documents'), request('/api/status')]);
  state.documents = library.documents;
  state.status = status;
  if (state.selected) state.selected = new Set([...state.selected].filter((id) => state.documents.some((doc) => doc.id === id)));
  $('connection').className = 'status-pill online';
  $('connection').replaceChildren(element('i'), document.createTextNode('Local · connected'));
  $('document-count').textContent = state.documents.length;
  $('chunk-count').textContent = `${formatNumber(status.chunks)} passages`;
  $('index-badge').textContent = status.dense_ready ? 'Ready' : 'Not built / stale';
  $('index-badge').className = status.dense_ready ? 'mini-badge teal' : 'mini-badge';
  $('index-description').textContent = status.dense_ready
    ? `Indexed with ${status.embedding_model}. Changes to documents require a rebuild.`
    : 'Lexical search works immediately. Dense and hybrid retrieval need optional dependencies, a model download and a current index.';
  renderLibrary();
  updateProviderNote();
}

function renderLibrary() {
  $('document-list').replaceChildren();
  $('library-empty').hidden = state.documents.length > 0;
  $('select-all').checked = state.selected === null;
  $('select-all').indeterminate = state.selected !== null && state.selected.size > 0;
  for (const doc of state.documents) {
    const row = element('div', 'document-row');
    const check = element('input');
    check.type = 'checkbox';
    check.checked = state.selected === null || state.selected.has(doc.id);
    check.setAttribute('aria-label', `Include ${doc.title} in search`);
    check.addEventListener('change', () => {
      if (state.selected === null) state.selected = new Set(state.documents.map((item) => item.id));
      if (check.checked) state.selected.add(doc.id); else state.selected.delete(doc.id);
      renderLibrary();
    });
    const title = element('button', 'document-title-button');
    title.type = 'button';
    title.title = `Inspect ${doc.title}`;
    title.append(element('strong', '', doc.title), element('small', '', `${formatNumber(doc.chunks)} passages · ${formatNumber(doc.characters)} characters`));
    title.addEventListener('click', () => inspect(doc.id));
    const remove = element('button', 'document-delete', '×');
    remove.type = 'button';
    remove.setAttribute('aria-label', `Remove ${doc.title}`);
    remove.addEventListener('click', () => {
      state.deleteId = doc.id;
      $('delete-name').textContent = doc.title;
      $('delete-dialog').showModal();
    });
    row.append(check, title, remove);
    $('document-list').append(row);
  }
  $('scope-note').textContent = state.selected === null ? 'Searching the whole library' : `Searching ${state.selected.size} selected document${state.selected.size === 1 ? '' : 's'}`;
}

function updateProviderNote() {
  const provider = $('provider').value;
  $('provider-note').className = provider === 'extractive' ? 'provider-note' : 'provider-note external';
  const notes = {
    extractive: 'Offline excerpts quote retrieved passages. No generation service is contacted.',
    ollama: 'Your question and retrieved passages are sent to your configured Ollama service. Its model and service must be available; configuration alone does not confirm that.',
    openai: 'External sharing: your question and retrieved passages will be sent to OpenAI. A server-side API key and model are required.',
    anthropic: 'External sharing: your question and retrieved passages will be sent to Anthropic. A server-side API key and model are required.'
  };
  let note = notes[provider];
  const config = state.status?.providers?.[provider];
  if (config && config.configured === false && provider !== 'extractive') note += ' This provider is not configured.';
  if ($('mode').value !== 'lexical') note += ' Semantic retrieval requires a current dense index.';
  if ($('rerank').checked) note += ' Reranking runs an optional cross-encoder model locally.';
  $('provider-note').textContent = note;
}

function clearAnswer() {
  $('answer-heading').textContent = 'Answer workspace';
  $('answer-empty').hidden = false;
  $('answer-content').hidden = true;
  $('sources-section').hidden = true;
}

async function ask(searchOnly = false) {
  if (state.busy) return;
  const query = $('question').value.trim();
  if (!query) { $('question').focus(); return; }
  if (!state.documents.length) { notify('Add a document or load the sample collection first.', true); return; }
  if (state.selected !== null && !state.selected.size) { notify('Choose at least one document, or select All documents.', true); return; }
  state.busy = true;
  notify('');
  $('ask-button').disabled = true;
  $('search-only').disabled = true;
  $('answer-empty').hidden = true;
  $('answer-content').hidden = true;
  $('sources-section').hidden = true;
  $('answer-loading').hidden = false;
  $('answer-panel').setAttribute('aria-busy', 'true');
  $('loading-text').textContent = searchOnly ? 'Retrieving relevant passages…' : 'Retrieving sources and preparing your answer…';
  const payload = {query, top_k: Number($('top-k').value), mode: $('mode').value, rerank: $('rerank').checked,
    document_ids: state.selected === null ? null : [...state.selected]};
  if (!searchOnly) payload.provider = $('provider').value;
  try {
    const result = await post(searchOnly ? '/api/search' : '/api/answer', payload);
    renderAnswer(result, searchOnly, payload);
  } catch (error) {
    $('answer-heading').textContent = 'The request could not be completed';
    $('answer-content').hidden = false;
    $('answer-type').textContent = 'Request error';
    $('answer-meta').textContent = '';
    $('answer-text').textContent = error.message;
    $('citation-links').replaceChildren();
    $('answer-caution').textContent = 'Adjust the settings or document selection and try again. RagKit does not silently switch models or retrieval modes.';
  } finally {
    state.busy = false;
    $('ask-button').disabled = false;
    $('search-only').disabled = false;
    $('answer-loading').hidden = true;
    $('answer-panel').setAttribute('aria-busy', 'false');
  }
}

function renderAnswer(result, searchOnly, settings) {
  const results = result.results || [];
  $('answer-content').hidden = false;
  $('answer-heading').textContent = result.abstained ? 'No supporting evidence found' : searchOnly ? 'Your retrieved passages' : 'An answer with sources';
  $('answer-type').textContent = searchOnly ? 'Retrieval only' : result.provider === 'extractive' ? 'Offline · extractive' : `${result.provider} · model answer`;
  $('answer-meta').textContent = `${settings.mode} retrieval${settings.rerank ? ' · reranked' : ''} · ${results.length} passage${results.length === 1 ? '' : 's'}`;
  $('answer-text').textContent = searchOnly
    ? (result.abstained ? 'There is not enough matching evidence in the selected documents. Try another question or add a relevant source.' : 'Review the matching passages below. No answer-generation service was called.')
    : result.answer;
  $('answer-caution').textContent = result.abstained ? 'No unsupported answer was generated.' : searchOnly ? 'Retrieval scores rank passages; they are not probabilities of factual correctness.' : result.provider === 'extractive'
    ? 'This is an extractive answer assembled from source quotes, not a model-generated explanation.'
    : 'Citation references are checked, but that does not establish that every model claim is true. Inspect the source context.';
  $('citation-links').replaceChildren();
  for (const [index, citation] of (result.citations || []).entries()) {
    const match = results.find((item) => item.document_id === citation.document_id && item.citation?.start === citation.start) || results[index];
    const title = citation.title || match?.title || 'Source';
    const button = element('button', 'citation-button', `${citation.id || `S${index + 1}`} · ${title}`);
    button.type = 'button';
    button.addEventListener('click', () => inspect(citation.document_id || match?.document_id, citation.start !== undefined ? citation : match?.citation));
    $('citation-links').append(button);
  }
  $('source-count').textContent = results.length;
  $('sources-section').hidden = !results.length;
  $('sources-list').replaceChildren();
  results.forEach((item, index) => {
    const card = element('article', 'source-card');
    const top = element('div', 'source-card-top');
    top.append(element('span', 'mini-badge teal', `S${index + 1}`), element('strong', '', item.title));
    const quote = element('blockquote', '', item.citation?.quote || item.text);
    const footer = element('div', 'source-card-footer');
    const score = Number.isFinite(item.score) ? item.score.toFixed(3) : '—';
    footer.append(element('span', '', `Score ${score} · chars ${item.citation?.start ?? '?'}–${item.citation?.end ?? '?'}`));
    const inspectButton = element('button', 'text-button', 'Inspect source ↗');
    inspectButton.type = 'button';
    inspectButton.addEventListener('click', () => inspect(item.document_id, item.citation));
    footer.append(inspectButton);
    card.append(top, quote, footer);
    $('sources-list').append(card);
  });
}

async function inspect(id, citation = null) {
  if (!id) return;
  $('source-title').textContent = 'Loading source…';
  $('source-location').textContent = '';
  $('source-original').textContent = 'Loading the stored original document…';
  $('source-dialog').showModal();
  try {
    const doc = await request(`/api/documents/${encodeURIComponent(id)}`);
    $('source-title').textContent = doc.title;
    $('source-location').textContent = `${doc.source || 'Local document'} · ${formatNumber(doc.characters)} characters`;
    const container = $('source-original');
    container.replaceChildren();
    // Python citations use Unicode code-point offsets, not JavaScript UTF-16 units.
    const characters = Array.from(doc.text);
    // An explicit-ID import can replace a document after an answer was retrieved.
    // Never present the new text at an old offset as evidence for that answer.
    const matches = citation && Number.isInteger(citation.start) && Number.isInteger(citation.end)
      && citation.start >= 0 && citation.end > citation.start && citation.end <= characters.length
      && typeof citation.quote === 'string'
      && characters.slice(citation.start, citation.end).join('') === citation.quote;
    if (matches) {
      $('source-location').textContent += ` · highlighted characters ${citation.start}–${citation.end}`;
      const mark = element('mark', '', citation.quote);
      container.append(document.createTextNode(characters.slice(0, citation.start).join('')), mark, document.createTextNode(characters.slice(citation.end).join('')));
      mark.scrollIntoView({block: 'center'});
    } else {
      container.textContent = doc.text;
      if (citation) $('source-location').textContent += ' · This document no longer matches this citation. Run your search again; no text is highlighted.';
    }
  } catch (error) { $('source-original').textContent = error.message; }
}

function openAdd() { $('add-error').hidden = true; $('upload-progress').textContent = ''; $('add-dialog').showModal(); }
function addError(message) { $('add-error').textContent = message; $('add-error').hidden = !message; }

async function uploadFiles(files) {
  if (!files.length) return;
  addError('');
  $('file-input').disabled = true;
  let added = 0;
  try {
    for (const [index, file] of [...files].entries()) {
      if (file.size > 8 * 1024 * 1024) throw new Error(`${file.name} exceeds the 8 MB limit.`);
      $('upload-progress').textContent = `Uploading ${index + 1} of ${files.length}: ${file.name}`;
      const body = new FormData(); body.append('file', file);
      const result = await request('/api/upload', {method: 'POST', body});
      added += result.documents.length;
    }
    await refresh();
    clearAnswer();
    $('add-dialog').close();
    notify(`${added} document${added === 1 ? '' : 's'} added or refreshed in your library.`);
  } catch (error) { addError(`${error.message}${added ? ` ${added} earlier document(s) were saved.` : ''}`); await refresh().catch(() => {}); }
  finally { $('file-input').disabled = false; $('file-input').value = ''; $('upload-progress').textContent = ''; }
}

$('question-form').addEventListener('submit', (event) => { event.preventDefault(); ask(); });
$('search-only').addEventListener('click', () => ask(true));
$('question').addEventListener('keydown', (event) => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); ask(); } });
['provider', 'mode', 'rerank'].forEach((id) => $(id).addEventListener('change', updateProviderNote));
$('select-all').addEventListener('change', () => { state.selected = $('select-all').checked ? null : new Set(); renderLibrary(); });
['add-documents', 'upload-shortcut'].forEach((id) => $(id).addEventListener('click', openAdd));
document.querySelectorAll('[data-close]').forEach((button) => button.addEventListener('click', () => $(button.dataset.close).close()));
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => { $('question').value = button.dataset.question; $('question').focus(); }));
['upload', 'paste'].forEach((tab) => $(`${tab}-tab`).addEventListener('click', () => {
  ['upload', 'paste'].forEach((name) => { $(`${name}-tab`).setAttribute('aria-selected', String(name === tab)); $(`${name}-tab-panel`).hidden = name !== tab; });
  addError('');
}));
$('file-input').addEventListener('change', (event) => uploadFiles(event.target.files));
['dragenter', 'dragover'].forEach((name) => $('drop-zone').addEventListener(name, (event) => { event.preventDefault(); $('drop-zone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => $('drop-zone').addEventListener(name, (event) => { event.preventDefault(); $('drop-zone').classList.remove('dragging'); }));
$('drop-zone').addEventListener('drop', (event) => { if (!$('file-input').disabled) uploadFiles(event.dataTransfer.files); });
$('paste-tab-panel').addEventListener('submit', async (event) => {
  event.preventDefault(); addError(''); $('save-paste').disabled = true;
  try {
    await post('/api/documents', {title: $('paste-title').value, text: $('paste-text').value, source: $('paste-source').value || null});
    await refresh(); clearAnswer(); $('paste-tab-panel').reset(); $('add-dialog').close(); notify('Document added to your library.');
  } catch (error) { addError(error.message); }
  finally { $('save-paste').disabled = false; }
});
$('load-sample').addEventListener('click', async () => {
  $('load-sample').disabled = true;
  try { const result = await post('/api/sample'); await refresh(); clearAnswer(); $('sample-prompts').hidden = false; notify(`Loaded ${result.documents.length} synthetic sample documents. They contain demonstration data, not company records.`); }
  catch (error) { notify(error.message, true); }
  finally { $('load-sample').disabled = false; }
});
$('confirm-delete').addEventListener('click', async () => {
  $('confirm-delete').disabled = true;
  try { await request(`/api/documents/${encodeURIComponent(state.deleteId)}`, {method: 'DELETE'}); $('delete-dialog').close(); await refresh(); clearAnswer(); notify('Document removed from the local workspace.'); }
  catch (error) { $('delete-dialog').close(); notify(error.message, true); }
  finally { $('confirm-delete').disabled = false; }
});
$('build-index').addEventListener('click', async () => {
  if (!state.documents.length) { notify('Add documents before building a semantic index.', true); return; }
  $('build-index').disabled = true; $('build-index').textContent = 'Building index…';
  notify('Building the semantic index. The configured embedding model may need to download on first use.');
  try { const result = await post('/api/index'); await refresh(); notify(`Semantic index built for ${result.chunks} passages with ${result.model}.`); }
  catch (error) { notify(error.message, true); }
  finally { $('build-index').disabled = false; $('build-index').textContent = 'Build semantic index ↗'; }
});
refresh().catch((error) => { $('connection').className = 'status-pill failed'; $('connection').replaceChildren(element('i'), document.createTextNode('Unavailable')); notify(`Could not connect to the workspace. ${error.message}`, true); });
