from pathlib import Path
from collections import defaultdict
import re, hashlib, json, stat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/项目整理'
NAV = 'outputs/项目整理'
START = '00-AI Infra 项目总览'
MARK = '\n\n%% 项目关联导航：开始 %%'
EXCLUDE = {'copilot','node_modules','__pycache__','outputs'}
files = sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(s.startswith('.') or s in EXCLUDE for s in p.relative_to(ROOT).parts))
files = [p for p in files if p != ROOT/(START+'.md')]
notes = [p for p in files if p.suffix.lower() == '.md' and p.name != 'AGENTS.md']
raw = {p: p.read_bytes() for p in notes}
manifest_path=OUT/'original_manifest.json'
if manifest_path.exists():
    previous=json.loads(manifest_path.read_text(encoding='utf-8'))
    for p in notes:
        record=previous[p.relative_to(ROOT).as_posix()]
        raw[p]=raw[p][:record['bytes']]
        assert hashlib.sha256(raw[p]).hexdigest()==record['sha256']
texts = {p: b.decode('utf-8-sig') for p,b in raw.items()}
def rel(p): return p.relative_to(ROOT).as_posix()
def link(p, label=None):
    s = rel(p) if isinstance(p,Path) else p
    if s.endswith('.md'): s=s[:-3]
    return '[['+s+'|'+(label or Path(s).name).replace('|','-').replace(']','')+']]'
def code(p):
    return any(s in ('code','gemm-project','nano_vllm_5070ti_starter') for s in p.relative_to(ROOT).parts)
def group(p):
    a = p.relative_to(ROOT).parts
    if a[0]=='项目_2026_722': return a[1]
    if a[0]=='实习准备_2026_831': return '面试记录' if a[1]=='面试记录' else '面试准备'
    if a[0]=='寒武纪实习':
        return '实习代码与资源' if a[1]=='code' else ('实习日记' if a[1]=='everyday' else '实习'+a[2])
    return a[0]
descs={
 'AI infra学习路线':'规划入口：明确学习目标，再把课程、推理项目和算子项目放进同一条能力路线。',
 'pytorch学习':'理解数据、张量、模型训练与推理，为模型结构和推理框架打基础。',
 'transformer':'理解模型内部的数据流，再与 nano-vLLM 的模型执行流程对照。',
 'nano_vllm':'先读整体调用关系，再沿请求入口、调度、缓存管理、模型执行、采样逐文件阅读。',
 'nano-vllm-qwen3.6':'围绕模型适配的增量阅读；笔记同时涉及 Qwen3.5 和 Qwen3.6，具体版本以各篇正文为准。',
 'nano-kvllm':'理解 KV Cache 压缩方案，以及它对缓存管理和 Attention 执行的影响。',
 'kvllm-qwen3.6融合':'连接 Qwen 适配与 KV 压缩两条项目线，阅读融合策略及任务书。',
 'vllm':'从生产推理框架的引擎、Worker、显存管理走向并行、编译、量化与性能优化。',
 'gemm算子':'从 CUDA 线程与存储层次进入 GEMM 分块计算，连接源码分析、性能测试与面试表达。',
 'nanovllm项目面试准备':'把实现细节转化为原理解释、实验依据和面试回答。',
 'leetcode':'按 Python 语法与题型复习算法，为手写题准备。',
 '提示词':'可复用的笔记生成与源码阅读方法；作为工作模板使用。',
 '面试准备':'从请求完整链路出发，复习 Qwen、KV 压缩、MTP、MoE 与项目问答。',
 '面试记录':'从真实问题回到知识点与项目实现，形成复习闭环。',
 '实习工程能力':'沿服务器操作、容器、服务启动、评测与代码协作流程查阅。',
 '实习项目能力':'连接 SGLang-MLU 插件结构、WeLM 模型、MoE 开发和运行环境。',
 '实习日记':'保留每日问题，从问题跳转到 MoE 或服务评测相关文档。',
 '实习代码与资源':'本地源码和附带文档的入口。这里只建立文件索引，具体调用关系参见源码阅读指南。',
}
groups=defaultdict(list)
for p in notes: groups[group(p)].append(p)
hubs={g:f'{NAV}/模块-{g}' for g in groups}
byname=defaultdict(list)
for p in notes: byname[p.stem].append(p)
def pick(name):
    ps=byname[name]
    knowledge=[p for p in ps if not code(p)]
    if knowledge: ps=knowledge
    assert len(ps)==1,(name,ps)
    return ps[0]

routes = [
 ('01-请求从输入到输出','先看模型结构，再看框架如何把请求组织成模型输入，最后练习完整口述。',[
 ('Decoder-only_LLM架构训练推理流程学习笔记','先理解 Token、Embedding、模型层与输出。'),
 ('nano-vllm源码结构与调用关系总览','建立框架文件与模型文件的连接。'),
 ('llm_engine.py_解析','从用户 API 进入推理引擎。'),
 ('sequence.py_解析','理解一个请求保存哪些状态。'),
 ('scheduler.py_解析','理解每轮如何选出执行请求。'),
 ('model_runner.py_解析','将请求状态转为模型需要的数据。'),
 ('10-nano-vLLM_Qwen3.5_一条请求完整调用链','把源码阅读收束为端到端表达。')]),
 ('02-KV缓存与调度','把分页管理、前缀复用、分块预填充和压缩放在一起对照；这些机制处理的问题不同。',[
 ('vLLM_KVCache_分页显存管理_课程学习笔记','先理解分页与物理缓存块。'),
 ('vLLM_请求与显存块映射关系_课程学习笔记','理解请求与 block 的映射。'),
 ('block_manager.py_解析','回到 nano-vLLM 的缓存管理代码。'),
 ('③Chunked Prefill、Continuous Batching 与 Prefix 前缀匹配','对照调度与缓存复用的原理问答。'),
 ('9-Qwen3.5_KV_Cache压缩_文件改动与完整链路','理解压缩给执行链路带来的变化。'),
 ('1-nano_qwen36_kv_compression_codex_strategy','查看模型适配与压缩融合的策略。')]),
 ('03-Qwen模型适配与MTP','从原版模型执行走向 Hybrid 状态管理、多模态和投机解码；按每篇记录的代码版本理解。',[
 ('qwen3.py_解析','原版模型结构入口。'),
 ('5-nano_vllm_qwen3_5_适配源码增量复盘','看适配新增了哪些状态和职责。'),
 ('6-Qwen3.5_GDN层内部处理流程详解','深入 GDN 层内部数据流。'),
 ('7-Qwen3.5_多模态适配文件与完整推理路径','扩展到图片与文本共同输入。'),
 ('8-nano_vLLM_Qwen3.5_MTP投机解码原型_文件与完整流程','理解草稿、验证与相关文件。'),
 ('推理项目面试QA','把实现细节整理成面试回答。')]),
 ('04-并行推理与MoE实习','先区分张量并行与专家并行，再进入 WeLM 结构和 MLU 适配。',[
 ('9.vLLM分布式推理_张量并行详解_学习笔记','学习权重切分与通信。'),
 ('14-Qwen3.5_TP4隐藏状态形状_权重切分_多卡通信完整流程','用 TP4 的张量形状串联执行。'),
 ('2-MoE模型从原理到推理系统完整学习指南','建立路由、专家计算与合并的基础。'),
 ('10.vLLM分布式推理_专家并行_学习笔记','对照专家并行的系统实现。'),
 ('WeLM模型结构详解','进入实习模型的具体结构。'),
 ('源码阅读与MoE开发指南','定位插件、模型与 MoE 开发文件。'),
 ('sglang_mlu_内部结构详解_20260909','补充内部模块关系；注意文中的快照日期。'),
 ('welm_moe_dual_stream方案','阅读双流方案，区分方案与已验证结果。')]),
 ('05-GEMM与推理性能','把算子存储层次、框架开销和服务性能指标分层理解。',[
 ('4-GEMM项目从零学习与源码阅读路线','先建立算子源码阅读顺序。'),
 ('16-GPU内部架构与CUDA_GEMM学习笔记','理解硬件执行与存储资源。'),
 ('17-GEMM矩阵划分到线程执行的完整路线详解','理解矩阵如何分到线程。'),
 ('18-GEMM项目数据完整流动详解','追踪 Global、Shared、Registers 数据流。'),
 ('算子项目面试QA','复习优化动机和测量口径。'),
 ('5.vLLM高级特性_CUDA_Graph详解_学习笔记','进入框架执行开销优化。'),
 ('14.vLLM性能分析和瓶颈定位_学习笔记','学习从指标和时间线定位瓶颈。')]),
 ('06-服务部署评测与协作','按环境准备、服务运行、评测分析、开发协作的工作顺序阅读。',[
 ('大模型推理服务器环境学习总结','先建立服务器与容器的整体认识。'),
 ('Docker从入门到AI服务器工作实战教程','查阅容器操作与目录挂载。'),
 ('Shell脚本与服务器常用命令实战教程','理解启动脚本与命令。'),
 ('EvalScope与模型服务启动常用参数汇总','查阅服务和评测参数。'),
 ('EvalScope使用_结果分析与内部评测原理','解释能力评测和性能压测的结果。'),
 ('git','查阅项目协作操作。'),
 ('大型项目_Patch_开发流程详解','理解补丁的制作和维护。')]),
 ('07-面试复盘闭环','从面试暴露的问题，回到原理和实现，最后形成可口述的项目介绍。',[
 ('三家公司面试核心问题汇总_精简版','先圈定需要补足的问题。'),
 ('20260812_面试复盘_最重点未答出的10个问题','定位具体薄弱点。'),
 ('文本到TokenID_30个面试原理问题深度学习','补足输入处理的原理追问。'),
 ('①FlashAttention','补足注意力算子的原理追问。'),
 ('推理项目面试QA','练习推理项目表述。'),
 ('算子项目面试QA','练习算子项目表述。'),
 ('自我介绍','收束为个人项目介绍。')]),
]
route_notes=defaultdict(list)
generated={}
def emit(name,body): generated[ROOT/(name+'.md')]=body
for title,intro,items in routes:
    name=f'{NAV}/专题-{title}'
    body=f'# {title}\n\n{link(START,"返回项目总览")}\n\n{intro}\n\n'
    for i,(stem,why) in enumerate(items,1):
        p=pick(stem)
        assert texts[p].strip(),p
        body+=f'{i}. {link(p)}：{why}\n'
        route_notes[p].append(name)
    emit(name,body)

for g,ps in groups.items():
    body=f'# {g} · 模块导航\n\n{link(START,"返回项目总览")}\n\n{descs.get(g,"按原目录查找材料。")}\n\n'
    associated=sorted({x for p in ps for x in route_notes[p]})
    if associated: body+='## 跨目录阅读路线\n\n'+'\n'.join('- '+link(x) for x in associated)+'\n\n'
    sub=defaultdict(list)
    for p in ps:sub[rel(p.parent)].append(p)
    for folder,children in sorted(sub.items()):
        body+=f'## {folder}\n\n'
        for p in children:
            hs=re.findall(r'^# (.+)',texts[p],re.M)
            label=re.sub(r'[`*]','',hs[0]).strip() if hs else p.stem
            label=label[:100]
            status='（空白笔记，待补充）' if not texts[p].strip() else ('（源码附带文档）' if code(p) else '')
            body+=f'- {link(p)}{status}'+(f' — {label}' if label!=p.stem else '')+'\n'
        body+='\n'
    emit(hubs[g],body)

body=f'# AI Infra 项目总览\n\n这里把课程、源码解析、项目扩展、面试复盘与实习文档连接成同一个知识入口。整理日期：2026-09-14。\n\n'
body+='## 建议学习顺序\n\nPyTorch／Transformer 基础 → nano-vLLM 请求与模型执行 → Qwen 适配、KV 压缩与 MTP → vLLM 并行及优化 → SGLang-MLU／WeLM 实习。GEMM 算子作为并行学习线，面试复盘用于发现和补足薄弱环节。\n\n'
body+='```mermaid\nflowchart TD\n A[PyTorch 与 Transformer] --> B[nano-vLLM 请求全流程]\n B --> C[Qwen 模型适配]\n B --> D[KV 缓存与调度]\n C --> E[KV 压缩融合与 MTP]\n D --> E\n B --> F[vLLM 并行与优化]\n F --> G[SGLang-MLU 与 WeLM MoE]\n H[CUDA 与 GEMM] --> F\n I[服务器与评测] --> G\n E --> J[面试表达与复盘]\n H --> J\n J --> B\n```\n\n'
body+='## 按问题进入\n\n'+'\n'.join('- '+link(f'{NAV}/专题-{t}')+'：'+i for t,i,_ in routes)+'\n\n'
body+='## 全部模块\n\n| 模块 | 文档数 | 用途 |\n| --- | --- | --- |\n'
for g,ps in groups.items():body+=f'| {link(hubs[g],g)} | {len(ps)} | {descs.get(g,"文件导航")} |\n'
body+=f'\n## 资源与维护\n\n- {link(NAV+"/附件与源码资源索引")}：查找代码、PDF、字幕、图片与压缩包。\n- {link(NAV+"/整理说明与校验报告")}：查看覆盖范围、重复文件和整理规则。\n\n新增笔记时，在所属模块导航中加一条链接，并在笔记末尾链接回模块；有明确前置或应用关系时，再补入对应专题。阅读路线表达知识联系，具体技术结论和实现状态以原文记录的版本与验证范围为准。\n'
emit(START,body)

resources=[p for p in files if p not in notes]
body=f'# 附件与源码资源索引\n\n{link(START,"返回项目总览")}\n\n按目录列出非笔记文件。源码、PDF、图片和压缩包仅按文件类型登记，未据文件名推断内部内容。源码阅读优先从 {link(pick("源码阅读与MoE开发指南"))} 和各模块已有说明开始。非 Markdown 文件能否直接打开取决于 Obsidian 的文件支持与系统关联。\n\n'
rs=defaultdict(list)
for p in resources:rs[rel(p.parent)].append(p)
for folder,ps in sorted(rs.items()):
    body+=f'## {folder}\n\n'+'\n'.join('- '+link(p) for p in ps)+'\n\n'
emit(NAV+'/附件与源码资源索引',body)

# Only append to knowledge notes. Preserve original bytes, frontmatter and all answer bodies.
updates={}
readonly=[]
for p in notes:
    if code(p) or not texts[p].strip():continue
    if p.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY:
        readonly.append(p)
        continue
    assert MARK not in texts[p],f'Navigation already exists: {p}'
    footer=MARK+'\n## 项目关联导航\n\n'+f'- 所属模块：{link(hubs[group(p)])}\n'
    for route in route_notes[p]:footer+='- 关联阅读：'+link(route)+'\n'
    footer+='\n%% 项目关联导航：结束 %%\n'
    updates[p]=raw[p]+footer.encode('utf-8')

duplicates=defaultdict(list)
for p,b in raw.items(): duplicates[hashlib.sha256(b).hexdigest()].append(p)
dup=[ps for ps in duplicates.values() if len(ps)>1]
report=f'# 整理说明与校验报告\n\n{link(START,"返回项目总览")}\n\n## 覆盖范围\n\n- 原始 Markdown 文档：{len(notes)} 篇；模块导航：{len(groups)} 篇；专题路线：{len(routes)} 条。\n- {len(updates)} 篇知识笔记末尾追加导航；空白笔记和源码目录附带文档保持原样。\n- 非笔记资源：{len(resources)} 个，按所在目录索引。\n- 排除 Obsidian 配置、隐藏目录、Copilot 技能、依赖目录及本次 outputs 产物。\n- 原目录和文件名保留；新增链接使用库内完整路径，避免同名笔记歧义。\n- 专题关系依据笔记正文中的主题、前置说明与文件职责整理；不代表重新核验了旧笔记中的技术结论。\n\n## 完全相同的原始文件\n\n以下按原始字节 SHA-256 分组，仅列出供辨认，保留各副本及已有链接。\n\n'
for ps in dup: report+='- '+'；'.join(link(p) for p in ps)+'\n'
if not dup:report+='未发现完全相同的文件。\n'
report+='\n## 空白笔记\n\n'
report+='\n'.join('- '+link(p) for p in notes if not texts[p].strip()) or '无。'
report+='\n\n## 验证结果\n\n全部新增链接目标均存在；每篇修改笔记的原始字节完整保留在文件开头；源码文档与其他原文件未改动。原有链接未做全库修复。原始文件摘要保存在同目录 original_manifest.json。\n'
report+='\n## 只读笔记\n\n以下笔记保留只读属性和原文，通过模块导航链接到它们：\n\n'+'\n'.join('- '+link(p) for p in readonly)+'\n'
emit(NAV+'/整理说明与校验报告',report)

if __name__=='__main__':
    if not manifest_path.exists():
        for p in generated:assert not p.exists(),f'Will not overwrite {p}'
    expected={rel(p) for p in files}|{rel(p) for p in generated}
    count=0
    for content in list(generated.values())+[v[len(raw[p]):].decode('utf-8') for p,v in updates.items()]:
        for target in re.findall(r'\[\[([^|\]]+)',content):
            assert target in expected or target+'.md' in expected,target
            count+=1
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'original_manifest.json').write_text(json.dumps({rel(p):{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'modified':p in updates} for p,b in raw.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    for p,t in generated.items():p.write_text(t,encoding='utf-8')
    for p,b in updates.items():p.write_bytes(b)
    for p,b in raw.items():
        now=p.read_bytes()
        assert now[:len(b)]==b
        if p not in updates:assert now==b
    print(json.dumps({'notes':len(notes),'updated':len(updates),'generated':len(generated),'groups':len(groups),'routes':len(routes),'resources':len(resources),'new_links_checked':count,'duplicate_groups':len(dup),'original_bytes_preserved':True},ensure_ascii=False))
