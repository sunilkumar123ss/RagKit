const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Exercise the browser entry point with its DOM surface; no third-party packages.
class Node {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.value = '';
  }
  set textContent(value) { this.children = []; this.text = value; }
  get textContent() { return this.children.length ? this.children.map(child => child.textContent).join('') : (this.text || ''); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ''; this.children = children; }
  addEventListener() {}
  setAttribute() {}
  showModal() {}
  scrollIntoView() { this.scrolled = true; }
}

function workspace(doc) {
  const nodes = new Map();
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, new Node());
    return nodes.get(id);
  };
  get('provider').value = 'extractive';
  get('mode').value = 'lexical';
  const context = vm.createContext({
    document: {
      getElementById: get,
      createElement: tag => new Node(tag),
      createTextNode: text => { const node = new Node('#text'); node.textContent = text; return node; },
      querySelectorAll: () => [],
    },
    fetch: async url => {
      let payload;
      if (url === '/api/documents') payload = {documents: []};
      else if (url === '/api/status') payload = {chunks: 0, dense_ready: false, providers: {}};
      else if (url === '/api/documents/policy') payload = doc;
      else throw new Error(`Unexpected request: ${url}`);
      return {ok: true, json: async () => payload};
    },
  });
  vm.runInContext(readFileSync(path.join(__dirname, '../ragkit/static/app.js'), 'utf8'), context);
  return {inspect: context.inspect, get};
}

function document(text) {
  return {id: 'policy', title: 'Policy', source: 'policy.md', text, characters: Array.from(text).length};
}

function marks(view) {
  return view.get('source-original').children.filter(node => node.tagName === 'mark');
}

test('inspection highlights the exact quote using Unicode code-point offsets', async () => {
  const view = workspace(document('👋 Café leave is 24 days.'));
  await view.inspect('policy', {start: 7, end: 24, quote: 'leave is 24 days.'});
  assert.equal(marks(view).length, 1);
  assert.equal(marks(view)[0].textContent, 'leave is 24 days.');
  assert.equal(marks(view)[0].scrolled, true);
  assert.equal(view.get('source-original').textContent, '👋 Café leave is 24 days.');
  assert.match(view.get('source-location').textContent, /highlighted characters 7–24/);
});

test('replacement text cannot become evidence for an old answer', async () => {
  const view = workspace(document('Leave is 10 days.'));
  await view.inspect('policy', {start: 0, end: 17, quote: 'Leave is 24 days.'});
  assert.equal(view.get('source-original').textContent, 'Leave is 10 days.');
  assert.equal(marks(view).length, 0);
  assert.match(view.get('source-location').textContent, /no longer matches this citation/i);
  assert.doesNotMatch(view.get('source-location').textContent, /highlighted characters/);
});

test('invalid or missing citation ranges do not produce misleading highlights', async () => {
  for (const citation of [
    {start: -1, end: 4, quote: 'Text'},
    {start: 0, end: 99, quote: 'Text'},
    {start: 2, end: 2, quote: ''},
    {start: 0, end: 4},
  ]) {
    const view = workspace(document('Text'));
    await view.inspect('policy', citation);
    assert.equal(view.get('source-original').textContent, 'Text');
    assert.equal(marks(view).length, 0);
    assert.match(view.get('source-location').textContent, /no longer matches this citation/i);
  }
});

test('inspection without a citation shows the full document', async () => {
  const view = workspace(document('Current original text.'));
  await view.inspect('policy');
  assert.equal(view.get('source-original').textContent, 'Current original text.');
  assert.doesNotMatch(view.get('source-location').textContent, /no longer matches|highlighted characters/i);
});
