"""Local context-compression viewer. No external calls or third-party scripts."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from .core import CounterBackend, compress, sentences

PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Token Budget Lab</title>
<style>body{max-width:1050px;margin:40px auto;font:16px system-ui;color:#17323b;background:#f6f8f8}textarea,input,select,button{font:inherit;padding:8px}textarea{width:95%;height:220px}input{width:65%}button{background:#167c80;color:white;border:0;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;background:white;padding:18px;border-radius:8px}.kept{background:#d5eee2}#source{line-height:1.8;background:white;padding:18px}</style>
<h1>Token Budget Lab</h1><p>观察问题相关的句子选择与邻域保留。预算按字符计数；本页面不调用大模型。</p>
<p><input id="query" value="水星有多少颗卫星？" aria-label="问题"></p>
<textarea id="context" aria-label="资料">金星表面温度很高。水星是距离太阳最近的行星。它没有天然卫星。木星有多颗卫星。天文学还研究恒星和星系。</textarea>
<p><select id="method"><option value="bm25">BM25</option><option value="bm25_neighbor">BM25 + 邻域</option><option value="head">Head</option></select>
<label>保留比例 <select id="ratio"><option>0.25</option><option selected>0.5</option><option>0.75</option></select></label> <button id="run">运行</button></p>
<p id="stats"></p><div class="grid"><div><h3>原文与保留句</h3><div id="source"></div></div><div><h3>发送的上下文</h3><pre id="selected"></pre></div></div>
<script>document.getElementById('run').onclick=async()=>{const body={};for(const id of ['query','context','method','ratio'])body[id]=document.getElementById(id).value;
try{const response=await fetch('/compress',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const r=await response.json();if(!response.ok)throw Error(r.error);
document.getElementById('stats').textContent=`原文 ${r.original} 字符 → ${r.selected.length} 字符；预算 ${r.budget}；耗时 ${r.elapsed_ms.toFixed(1)} ms`;
document.getElementById('selected').textContent=r.selected;const source=document.getElementById('source');source.replaceChildren();for(const s of r.sentences){const span=document.createElement('span');span.textContent=s+' ';if(r.selected.includes(s))span.className='kept';source.appendChild(span)}}catch(e){document.getElementById('stats').textContent=e.message}};</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data=PAGE.encode()
        self.send_response(200)
        self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        import time
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size <= 0 or size > 250000:
                raise ValueError('request must be between 1 and 250000 bytes')
            r=json.loads(self.rfile.read(size))
            ratio=float(r['ratio'])
            if not 0 < ratio <= 1:
                raise ValueError('ratio must be in (0, 1]')
            start=time.perf_counter()
            budget=int(len(r['context'])*ratio)
            selected=compress(r['query'],r['context'],budget,CounterBackend(),r['method'])
            data=dict(selected=selected,original=len(r['context']),budget=budget,
                      sentences=sentences(r['context']),elapsed_ms=(time.perf_counter()-start)*1000)
            status=200
        except (ValueError,KeyError,TypeError):
            data={'error':'输入无效，请检查问题、材料、方法和比例。'}
            status=400
        payload=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    print(f'Open http://127.0.0.1:{args.port}; Ctrl+C to stop.')
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':
    main()
