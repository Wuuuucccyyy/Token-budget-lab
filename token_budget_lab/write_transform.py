"""Host-declared numeric transforms bound to actual reads in one agent run."""
from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext

from .context_plugin import source_id
from .memory import encode


class WriteValueRejected(ValueError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail['reason'])

    def result(self):
        return dict(error='write_value_rejected', executed=False, written=False,
                    **self.detail, note='No write executed. Use the actual fresh source and the declared transform.')


def pointer_value(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('nonempty RFC6901 pointer required')
    for part in pointer[1:].split('/'):
        if re.search(r'~(?![01])', part):
            raise ValueError('invalid pointer escape')
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not re.fullmatch(r'0|[1-9][0-9]*', part):
                raise ValueError('invalid array index')
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise ValueError('pointer traverses scalar')
    return value


@dataclass(frozen=True)
class WriteTransform:
    """Trusted intent, supplied by the host; contains no expected source value.

    Selectors: pointer; unique_event (array + exact string key/value + field);
    text_record (text pointer + marker + key=value in a semicolon/newline record).
    The host must define this rule from user intent, never evaluation labels.
    """
    read_tool: str
    read_arguments: dict
    write_tool: str
    value_argument: str
    selector: dict
    operation: str = 'format_decimal'
    places: int = 2
    constant: int = 0
    fixed_arguments: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.operation not in {'format_decimal', 'add_integer'}:
            raise ValueError('unsupported transform')
        if type(self.places) is not int or not 0 <= self.places <= 6 or type(self.constant) is not int or abs(self.constant)>10**12:
            raise ValueError('invalid transform bounds')
        if self.operation == 'format_decimal' and self.constant != 0:
            raise ValueError('formatting cannot change numeric scale')
        kind = self.selector.get('kind')
        keys = {'pointer': {'kind','pointer'},
                'unique_event': {'kind','array','match_key','match_value','field'},
                'text_record': {'kind','pointer','marker','field'}}
        if kind not in keys or set(self.selector) != keys[kind]:
            raise ValueError('explicit supported selector required')
        if any(not isinstance(v,str) or not v or len(v)>256 for v in self.selector.values()):
            raise ValueError('bounded string selector required')
        if not isinstance(self.read_arguments,dict) or not isinstance(self.fixed_arguments,dict) or self.value_argument in self.fixed_arguments:
            raise ValueError('invalid tool arguments')


def numeric(value):
    # Avoid booleans, NaN, Infinity, huge exponents and unbounded conversions.
    if type(value) not in (str, int, float) or len(str(value)) > 128:
        raise ValueError('unsupported numeric value')
    if isinstance(value,str) and not re.fullmatch(r'[+-]?[0-9]+(?:\.[0-9]+)?',value):
        raise ValueError('plain decimal value required')
    result = Decimal(str(value))
    if not result.is_finite() or abs(result)>Decimal('1e18') or result.as_tuple().exponent < -18:
        raise ValueError('nonfinite or out-of-range value')
    return result


class FreshWriteGuard:
    """Validate before invocation; never execute or silently repair a write.

    No model-supplied provenance is trusted. Only the dispatcher's successful
    current-run handler returns can establish evidence. Recovery, history and
    stable-read cache echoes do not call observe(). Rules have fixed notices;
    expected values are computed only from actual reads and rejection feedback.
    """
    def __init__(self, rules):
        self.rules = copy.deepcopy(list(rules))
        if not self.rules or len({r.write_tool for r in self.rules}) != len(self.rules):
            raise ValueError('one nonempty rule per guarded write tool required')
        self.sources, self.events = {}, []

    def start(self, schemas, handlers):
        for rule in self.rules:
            for name in (rule.read_tool,rule.write_tool):
                if name not in schemas or name not in handlers or name.startswith(('context_','history_')):
                    raise ValueError('registered business handlers required')
            read_keys=schemas[rule.read_tool]['parameters'].get('properties',{})
            write_keys=schemas[rule.write_tool]['parameters'].get('properties',{})
            if set(rule.read_arguments)-set(read_keys) or {rule.value_argument,*rule.fixed_arguments}-set(write_keys):
                raise ValueError('transform arguments must match registered schemas')
        return FreshWriteGuard(self.rules)

    def tools(self, tools):
        return copy.deepcopy(tools)

    def notice(self):
        return ('[Write transform contract] The host checks configured writes against actual successful reads in THIS run. '
                'format_decimal preserves numeric magnitude (no division or cents conversion), with ROUND_HALF_EVEN; '
                'add_integer uses the stated integer offset. Unique matches are required. Rules: '+
                encode([asdict(r) for r in self.rules])+'. Rejected writes are not executed; correction uses the same step budget.')

    def before_invocation(self, name, arguments):
        for rule in self.rules:
            if name == rule.read_tool and arguments == rule.read_arguments:
                # A new failed read cannot fall back to a previous successful one.
                self.sources.pop(rule.write_tool, None)
            # Conservative invalidation before a source-changing write, even if
            # its handler partially mutates the file and then raises.
            if (name,rule.read_tool) in {('write_config','read_config'),('write_report','read_report')}:
                self.sources.pop(rule.write_tool,None)

    def observe(self, name, arguments, value, message, position, handler_returned):
        if not handler_returned or not isinstance(value,dict) or value.get('error') or value.get('success') is False:
            return
        for rule in self.rules:
            if name == rule.read_tool and arguments == rule.read_arguments:
                self.sources[rule.write_tool] = dict(source_id=source_id(position,message),
                    tool=name, arguments=copy.deepcopy(arguments), value=copy.deepcopy(value),
                    result_sha256=hashlib.sha256(encode(value).encode()).hexdigest())

    def _select(self, rule, source):
        s, value = rule.selector, source['value']
        if s['kind'] == 'pointer':
            return pointer_value(value,s['pointer']), s['pointer']
        if s['kind'] == 'unique_event':
            rows=pointer_value(value,s['array'])
            if not isinstance(rows,list):
                raise ValueError('selector array missing')
            matches=[(i,r) for i,r in enumerate(rows) if isinstance(r,dict)
                and type(r.get(s['match_key'])) is str and r[s['match_key']]==s['match_value']]
            if len(matches)!=1:
                raise ValueError('source_match_missing' if not matches else 'source_match_ambiguous')
            i,row=matches[0]
            escaped=s['field'].replace('~','~0').replace('/','~1')
            return row[s['field']],s['array']+'/'+str(i)+'/'+escaped
        text=pointer_value(value,s['pointer'])
        if not isinstance(text,str):
            raise ValueError('text source required')
        markers=list(re.finditer(r'(?<![\w.-])'+re.escape(s['marker'])+r'(?![\w.-])',text))
        if len(markers)!=1:
            raise ValueError('source_match_missing' if not markers else 'source_match_ambiguous')
        start=markers[0].end()
        end=re.search(r'[;\r\n]',text[start:])
        record=text[start:start+end.start()] if end else text[start:]
        pattern=r'(?<![\w])'+re.escape(s['field'])+r'\s*=\s*([+-]?[0-9]+(?:\.[0-9]+)?)(?![\w.])'
        fields=list(re.finditer(pattern,record))
        if len(fields)!=1:
            raise ValueError('text_field_missing_or_ambiguous')
        return fields[0].group(1), dict(pointer=s['pointer'],offset=start+fields[0].start(),end_offset=start+fields[0].end())

    def validate(self, name, arguments):
        rule=next((r for r in self.rules if r.write_tool==name),None)
        if rule is None:
            return arguments
        detail=dict(tool=name,operation=rule.operation,attempted_value=arguments.get(rule.value_argument))
        try:
            source=self.sources.get(name)
            if source is None:
                raise ValueError('successful_fresh_read_required')
            detail.update(source_id=source['source_id'],result_sha256=source['result_sha256'],read_tool=source['tool'],read_arguments=source['arguments'])
            raw,path=self._select(rule,source)
            number=numeric(raw)
            detail.update(source_value=str(raw),source_location=path)
            with localcontext() as context:
                context.prec=40
                if rule.operation=='format_decimal':
                    expected=format(number.quantize(Decimal(1).scaleb(-rule.places),rounding=ROUND_HALF_EVEN),'f')
                else:
                    if number!=number.to_integral_value():
                        raise ValueError('integer_source_required')
                    expected=int(number)+rule.constant
            detail['expected_value']=expected
            if any(arguments.get(k)!=v for k,v in rule.fixed_arguments.items()):
                raise ValueError('fixed_arguments_do_not_match_task')
            candidate=arguments.get(rule.value_argument)
            if type(candidate) is not type(expected) or candidate!=expected:
                raise ValueError('write_value_does_not_match_fresh_transform')
        except (ValueError,KeyError,IndexError,TypeError,InvalidOperation) as exc:
            reason=str(exc) if type(exc) is ValueError else 'source_field_invalid'
            detail['reason']=reason
            self.events.append(dict(accepted=False,**copy.deepcopy(detail)))
            raise WriteValueRejected(detail) from None
        self.events.append(dict(accepted=True,**copy.deepcopy(detail)))
        return arguments

    def state(self):
        return dict(rules=[asdict(r) for r in self.rules],events=copy.deepcopy(self.events),
                    accepted=sum(e['accepted'] for e in self.events),rejected=sum(not e['accepted'] for e in self.events))
