/** Joins class names, skipping false/null/undefined: cx('seg', on && 'on'). */
export const cx = (...names: (string | false | null | undefined)[]) => names.filter(Boolean).join(' ')
