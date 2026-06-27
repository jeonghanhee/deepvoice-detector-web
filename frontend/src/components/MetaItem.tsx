interface MetaItemProps {
  label: string;
  value: string;
}

export function MetaItem({ label, value }: MetaItemProps) {
  return (
    <div className="meta-item">
      <div className="meta-key">{label}</div>
      <div className="meta-val">{value}</div>
    </div>
  );
}
