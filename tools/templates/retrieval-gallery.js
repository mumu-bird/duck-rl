// All episode data is embedded in the page so it also works from a local file.
const el = (id) => document.getElementById(id);
const data = JSON.parse(el('records').textContent);
const versions = {v1: 'V1 原策略', v3: 'V3 平滑起身', v4: 'V4 重新定位', v6: 'V6 贴地探入'};
const [panel, item, newer, older, filter, reasonFilter] =
  ['panel', 'item', 'newer', 'older', 'filter', 'reason'].map(el);
let list = [];
let index = 0;

function makeLink(url, text) {
  const link = document.createElement('a');
  link.href = url;
  link.textContent = text;
  return link;
}

for (const entry of JSON.parse(el('extra-records').textContent)) {
  const title = `${entry.group} · ${entry.name} ${entry.seed} · ${entry.reason}`;
  el('extras').append(makeLink(entry.url, title), document.createElement('br'));
}

function setOptions(select, choices, keep) {
  select.replaceChildren(...choices.map(([value, label]) => new Option(label, value)));
  if (choices.some(([key]) => key === keep)) select.value = keep;
}

function loadPanel() {
  const rows = data.filter((row) => row.panel === panel.value);
  const items = new Map(rows.map((row) => [row.id, row.name]));
  setOptions(item, [['all', '所有物体'], ...items], 'all');
  const available = Object.keys(rows[0].versions).sort();
  const options = available.map((key) => [key, versions[key] || key]);
  setOptions(older, options);
  setOptions(newer, [...options].reverse());
  reasonFilter.value = 'all';
}

function displayPlayer(id, row, version) {
  const player = el(id);
  const result = row.versions[version];
  const status = document.createElement('span');
  status.className = result.ok ? 'ok' : 'bad';
  status.textContent = result.reason;
  player.querySelector('header').replaceChildren(versions[version], status);
  const movie = player.querySelector('video');
  movie.pause();
  movie.setAttribute('src', result.url);
  const stats = `共尝试 ${result.attempts} 次 · 结束时离起点 ${result.home} cm`;
  player.querySelector('footer').replaceChildren(
    stats, makeLink(result.url, '单独看 MP4'), makeLink(result.log, '逐次记录')
  );
}

function show(position) {
  if (!list.length) return;
  index = (position + list.length) % list.length;
  const row = list[index];
  el('title').textContent = `${row.name} · 投掷 ${row.seed}`;
  displayPlayer('left', row, older.value);
  displayPlayer('right', row, newer.value);
  const state = new URLSearchParams({p: panel.value, o: row.id, s: row.seed, a: older.value, b: newer.value});
  el('permalink').href = '#' + state;
  Array.from(el('rows').rows).forEach((tr, i) => tr.classList.toggle('selected', i === index));
}

function populateTable() {
  const body = el('rows');
  body.replaceChildren();
  list.forEach((row, i) => {
    const tr = body.insertRow();
    const values = [row.name, row.seed, row.versions[older.value].reason, row.versions[newer.value].reason];
    values.forEach((value) => tr.insertCell().textContent = value);
    const button = document.createElement('button');
    button.textContent = '查看';
    button.onclick = () => show(i);
    tr.insertCell().append(button);
  });
}

function refresh(prefer) {
  const rows = data.filter((r) => r.panel === panel.value && (item.value === 'all' || r.id === item.value));
  const old = (r) => r.versions[older.value];
  const now = (r) => r.versions[newer.value];
  const reasons = [...new Set(rows.filter((r) => !now(r).ok).map((r) => now(r).reason))].sort();
  setOptions(reasonFilter, [['all', '所有阶段'], ...reasons.map((r) => [r, r])], reasonFilter.value);
  const results = {
    all: () => true,
    gain: (r) => now(r).ok && !old(r).ok,
    loss: (r) => !now(r).ok && old(r).ok,
    ok: (r) => now(r).ok,
    bad: (r) => !now(r).ok
  };
  list = rows.filter((r) => results[filter.value](r) && (reasonFilter.value === 'all' || now(r).reason === reasonFilter.value));
  const oldPass = rows.filter((r) => old(r).ok).length;
  const newPass = rows.filter((r) => now(r).ok).length;
  el('counts').textContent = `同组 ${rows.length} 次投掷　对照通过 ${oldPass} 次　比较版本通过 ${newPass} 次　当前筛选 ${list.length} 次`;
  populateTable();
  el('empty').hidden = list.length > 0;
  el('stage').hidden = list.length === 0;
  if (!list.length) {
    pausePlayers();
    return;
  }
  const match = prefer ? list.findIndex(prefer) : -1;
  show(match < 0 ? 0 : match);
}

function startup() {
  const query = new URLSearchParams(location.hash.slice(1));
  if (data.some((r) => r.panel === query.get('p'))) panel.value = query.get('p');
  loadPanel();
  filter.value = 'all';
  for (const [select, key] of [[older, 'a'], [newer, 'b']]) {
    if (Array.from(select.options).some((o) => o.value === query.get(key))) select.value = query.get(key);
  }
  const prefer = query.has('s')
    ? (r) => r.seed === Number(query.get('s')) && r.id === query.get('o')
    : (r) => r.versions[newer.value].ok && !r.versions[older.value].ok;
  refresh(prefer);
}

function pausePlayers() {
  document.querySelectorAll('video').forEach((movie) => movie.pause());
}

panel.onchange = () => {loadPanel(); refresh();};
[item, newer, older, filter, reasonFilter].forEach((select) => select.onchange = () => refresh());
el('play').onclick = () => {
  document.querySelectorAll('video').forEach((movie) => {
    movie.currentTime = 0;
    movie.play().catch(() => { el('counts').textContent = '请点击视频内的播放键。'; });
  });
};
el('pause').onclick = pausePlayers;
el('prev').onclick = () => show(index - 1);
el('next').onclick = () => show(index + 1);
window.addEventListener('hashchange', startup);
startup();
