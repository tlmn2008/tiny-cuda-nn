#!/usr/bin/env python3
"""Failure-Gate workaround attempt: neutralize NV-PTX inline asm in CUTLASS
arch/memory.h with generic C++ for ivcore11 (Iluvatar CoreX). See notes in the
migration session log. global_load/store use real pointers (correct); shared_
load/store reinterpret the truncated uint32_t token (probe of the address gap).
"""
import re, pathlib

p = pathlib.Path("dependencies/cutlass/include/cutlass/arch/memory.h")
s = p.read_text()

def find_asm_blocks(text):
    blocks = []
    for m in re.finditer(r'asm volatile\(', text):
        i = m.end()  # position just after '('
        depth = 1
        instr = False
        esc = False
        while i < len(text) and depth > 0:
            c = text[i]
            if instr:
                if esc: esc = False
                elif c == '\\': esc = True
                elif c == '"': instr = False
            else:
                if c == '"': instr = True
                elif c == '(': depth += 1
                elif c == ')': depth -= 1
            i += 1
        # expect trailing ';'
        while i < len(text) and text[i] in ' \t\n': i += 1
        if i < len(text) and text[i] == ';': i += 1
        blocks.append((m.start(), i))
    return blocks

def classify(ctx):
    cands = {
        'gl': ctx.rfind('struct global_load'),
        'gs': ctx.rfind('struct global_store'),
        'sl': ctx.rfind('shared_load'),
        'ss': ctx.rfind('shared_store'),
    }
    return max(cands.items(), key=lambda kv: kv[1])

def shared_bytes(ctx):
    mm = re.findall(r'(?:shared_load|shared_store)<(\d+)>', ctx)
    return int(mm[-1]) if mm else 16

blocks = find_asm_blocks(s)
patched = 0
for st, en in reversed(blocks):
    asm = s[st:en]
    kind, pos = classify(s[max(0, st-600):st])
    if pos == -1:
        continue
    if kind == 'gl':
        generic = "if (pred_guard) { D = *reinterpret_cast<AccessType const *>(ptr); }"
    elif kind == 'gs':
        generic = "if (pred_guard) { *reinterpret_cast<AccessType *>(ptr) = D; }"
    elif kind == 'sl':
        nb = shared_bytes(s[max(0, st-200):st])
        generic = (f"struct AccessType_ivc {{ unsigned char _b[{nb}]; }};\n"
                   f"    *reinterpret_cast<AccessType_ivc *>(dst) = "
                   f"*reinterpret_cast<const AccessType_ivc *>(static_cast<uintptr_t>(ptr));")
    else:
        nb = shared_bytes(s[max(0, st-200):st])
        generic = (f"struct AccessType_ivc {{ unsigned char _b[{nb}]; }};\n"
                   f"    *reinterpret_cast<AccessType_ivc *>(static_cast<uintptr_t>(ptr)) = "
                   f"*reinterpret_cast<const AccessType_ivc *>(src);")
    wrapped = ("#if defined(__ILUVATAR__)\n    " + generic + "\n#else\n" + asm + "\n#endif")
    s = s[:st] + wrapped + s[en:]
    patched += 1

p.write_text(s)
print(f"patched {patched} asm blocks")
