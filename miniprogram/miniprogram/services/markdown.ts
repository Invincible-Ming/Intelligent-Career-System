// Model/user text is escaped first. No remote images, executable HTML or external
// navigation is accepted from generated content.
function escape(value:string):string{return value.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
function inline(value:string):string{return escape(value).replace(/\[([^\]]+)\]\([^)]*\)/g,'$1').replace(/`([^`]+)`/g,'<code style="font-family:monospace;background:rgba(128,128,128,.12);padding:2px 5px;border-radius:4px">$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')}
export function markdown(text:string):string{
 const output:string[]=[],code:string[]=[];let fenced=false
 for(const line of text.split('\n')){
  if(line.trim().startsWith('```')){if(fenced){output.push('<pre style="white-space:pre-wrap;word-break:break-all;background:rgba(128,128,128,.12);padding:14px;border-radius:10px;font-size:13px">'+escape(code.join('\n'))+'</pre>');code.length=0}fenced=!fenced;continue}
  if(fenced){code.push(line);continue}
  if(!line.trim()){output.push('<div style="height:8px"></div>');continue}
  const heading=line.match(/^(#{1,4})\s+(.+)$/)
  if(heading){output.push('<div style="font-weight:650;font-size:'+(heading[1].length===1?'20':'17')+'px;margin:18px 0 10px">'+inline(heading[2])+'</div>');continue}
  const list=line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)$/)
  if(list){output.push('<div style="margin:6px 0;padding-left:12px">• '+inline(list[1])+'</div>');continue}
  if(/^\s*---+\s*$/.test(line)){output.push('<hr/>');continue}
  output.push('<div style="margin:5px 0;word-break:break-word">'+inline(line.replace(/^>\s?/,''))+'</div>')
 }
 if(code.length)output.push('<pre style="white-space:pre-wrap;word-break:break-all">'+escape(code.join('\n'))+'</pre>')
 return output.join('')
}
