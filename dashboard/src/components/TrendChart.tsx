export const TrendChart = () => {
  return (
    <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
      <div className="flex justify-between items-baseline mb-[18px]">
        <div className="text-[14.5px] font-semibold text-text-hi font-heading">Success rate trend</div>
        <div className="text-[12px] text-text-dim">Avg across all builds</div>
      </div>
      
      <svg viewBox="0 0 520 160" width="100%" height="160" role="img" aria-label="Line chart showing average task success rate">
        <defs>
          <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00F0FF" stopOpacity="0.18"/>
            <stop offset="100%" stopColor="#00F0FF" stopOpacity="0"/>
          </linearGradient>
          <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#00F0FF"/>
            <stop offset="100%" stopColor="#BD00FF"/>
          </linearGradient>
        </defs>
        <path d="M0,110 L83,96 L166,86 L250,78 L333,58 L416,44 L500,34 L500,160 L0,160 Z" fill="url(#area)"/>
        <polyline points="0,110 83,96 166,86 250,78 333,58 416,44 500,34" fill="none" stroke="url(#line)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
        <g fill="#00F0FF">
          <circle cx="0" cy="110" r="2.5"/>
          <circle cx="83" cy="96" r="2.5"/>
          <circle cx="166" cy="86" r="2.5"/>
          <circle cx="250" cy="78" r="2.5"/>
          <circle cx="333" cy="58" r="2.5"/>
          <circle cx="416" cy="44" r="2.5"/>
        </g>
        <circle cx="500" cy="34" r="3.5" fill="#BD00FF"/>
      </svg>
      
      <div className="flex gap-4 text-[12px] text-text-dim mt-3.5">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-[2px] bg-cyan" /> Mon–Sat
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-[2px] bg-magenta" /> Today, 85.1%
        </div>
      </div>
    </div>
  );
};
