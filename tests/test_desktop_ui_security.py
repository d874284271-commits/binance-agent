"""Exercise the shipped UI's text boundaries and asynchronous result selection."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest


def run_ui(script: str):
    executable = shutil.which("node")
    if not executable:
        pytest.skip("Node required for JavaScript behavior checks")
    harness = r"""
const fs=require('fs'), vm=require('vm');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.value='';this.hidden=false;this.checked=false;this.listeners={};this.dataset={};this._text='';}
 set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
 setAttribute(k,v){this.attrs[k]=String(v);if(k==='value')this.value=String(v);}
 removeAttribute(k){delete this.attrs[k];} getAttribute(k){return this.attrs[k];}
 append(c){this.children.push(c);} replaceChildren(...c){this._text='';this.children=c;}
 addEventListener(k,f){this.listeners[k]=f;} focus(){} close(){} showModal(){}
}
const elements=new Map(), storage=new Map();
function get(id){if(!elements.has(id))elements.set(id,new Element('div'));return elements.get(id);}
const context=vm.createContext({console,URL,Blob,Node:Element,document:{getElementById:get,createElement:t=>new Element(t),createTextNode:t=>{const n=new Element('#text');n.textContent=t;return n;},querySelectorAll:()=>[]},window:{addEventListener(){}},localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},crypto:{randomUUID:()=> 'uuid'},setTimeout:()=>1,clearTimeout(){},navigator:{},finish:v=>process.stdout.write(JSON.stringify(v))});
vm.runInContext(fs.readFileSync('src/binance_agent/desktop/ui/app.js','utf8').split('\n(async function init()')[0],context);
vm.runInContext('(async()=>{'+process.argv[1]+'})()',context).catch(e=>{console.error(e);process.exitCode=1});
"""
    result = subprocess.run(
        [executable, "-e", harness, script], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_external_report_text_cannot_create_markup_or_script_links():
    result = run_ui(r"""
const payload='<img src=x onerror=alert(1)>', bad='javascript:alert(1)';
renderReport({summary:payload,analysis_at:null,completeness:'partial',risks:[payload],recommendations:[payload],basis:[{statement:payload,quote:payload}],sources:[{title:payload,url:bad}],gaps:[payload],disclaimer:payload});
const tags=[];function walk(n){tags.push({tag:n.tag,attrs:n.attrs});n.children.forEach(walk);}walk($('report-content'));
finish({text:$('report-content').textContent,tags,unknown:timeText(null)});
""")
    assert result["text"].count("<img src=x onerror=alert(1)>") >= 6
    assert not {"img", "script", "iframe", "object"}.intersection(x["tag"] for x in result["tags"])
    assert not any(x["attrs"].get("href", "").startswith("javascript:") for x in result["tags"])
    assert result["unknown"] == "时点未知"


def test_late_result_cannot_replace_selected_historical_version():
    result = run_ui(r"""
const resolvers={}, shown=[];
api=path=>new Promise(resolve=>resolvers[path]=resolve);
renderReport=(report,id)=>shown.push([id,report.summary]);
state.session={runs:[{run_id:'old',status:'completed'},{run_id:'new',status:'completed'}]};
state.selected='old';const first=loadSelectedReport();
state.selected='new';const second=loadSelectedReport();
resolvers['/runs/new/report']({report:{summary:'new report'}});await second;
resolvers['/runs/old/report']({report:{summary:'late old report'}});await first;
finish(shown);
""")
    assert result == [["new", "new report"]]


def test_mode_switch_preserves_separate_drafts_without_starting_research():
    result = run_ui(r"""
state.catalog.fapi=[{symbol:'BTCUSDT',base_asset:'BTC',quote_asset:'USDT',name:'Bitcoin'}];
setMode('report',false);$('symbol').value='BTCUSDT';
setMode('topic');$('topic-question').value='original topic';$('link-asset').checked=false;
setMode('report');setMode('topic');
finish({question:$('topic-question').value,linked:$('link-asset').checked,marketHidden:$('asset-fields').hidden,report:state.drafts.report,topic:state.drafts.topic});
""")
    assert result["question"] == "original topic" and result["marketHidden"]
    assert "question" not in result["report"]
    assert not result["linked"]


def test_provider_save_failure_preserves_inputs_and_visible_detail():
    result = run_ui(r"""
state.config={llm:{endpoints:[{name:'a',base_url:'https://old.example/v1',model:'m'}]}};
$('model-service').value='__new__';$('endpoint-name').value='b';$('endpoint-url').value='https://new.example/v1';$('endpoint-model').value='m';$('endpoint-key').value='';
api=async()=>{throw new Error('需要为新地址重新提供密钥');};
await saveModel({preventDefault(){}});
finish({message:$('settings-message').textContent,url:$('endpoint-url').value,old:state.config.llm.endpoints[0].base_url});
""")
    assert "重新提供" in result["message"]
    assert result["url"] == "https://new.example/v1" and result["old"] == "https://old.example/v1"


def test_late_legacy_report_cannot_replace_new_session_or_export_selection():
    result = run_ui(r"""
let release;const shown=[];
api=path=>path==='/runs/legacy/report'?new Promise(resolve=>release=resolve):Promise.resolve(path==='/runs/new/report'?{report:{summary:'current report'}}:{session_id:'current',runs:[{run_id:'new',status:'completed'}]});
renderReport=(report,id)=>shown.push(id);renderSession=()=>{};subscribe=()=>{};
const pending=openLegacyReport('legacy');
await openSession('current');
release({report:{title:'old',summary:'old report'}});await pending;
finish({shown,selected:state.selected,session:state.session.session_id,report:state.report.summary});
""")
    assert result == {
        "shown": ["new"],
        "selected": "new",
        "session": "current",
        "report": "current report",
    }
