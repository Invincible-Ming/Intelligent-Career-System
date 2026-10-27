// Small display helpers shared by the mobile screens. No data leaves the device.
export function dateLabel(value?: string): string {
  if (!value) return ''
  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return ''
  const two = (n:number):string => String(n).padStart(2,'0')
  return date.getFullYear() + '/' + two(date.getMonth()+1) + '/' + two(date.getDate()) + ' ' + two(date.getHours()) + ':' + two(date.getMinutes())
}
export function fileExtension(filename:string):string {
  const extension = filename.includes('.') ? filename.split('.').pop() || '' : ''
  return extension.length > 0 && extension.length <= 5 ? extension.toUpperCase() : 'FILE'
}
