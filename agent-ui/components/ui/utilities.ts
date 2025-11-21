// Utility: merges Tailwind classes
export function cn(...inputs: any[]) {
  return inputs
    .flat(Infinity)
    .filter(Boolean)
    .join(" ");
}