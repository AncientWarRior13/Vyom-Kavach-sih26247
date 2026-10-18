import React, { useState } from 'react';

export interface Column<T> {
  key: string;
  header: React.ReactNode;
  width?: string;
  align?: 'left' | 'center' | 'right';
  sortable?: boolean;
  render?: (row: T, index: number) => React.ReactNode;
}

export interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyField: keyof T | ((row: T) => string);
  onRowClick?: (row: T) => void;
  selectedKey?: string;
  className?: string;
  emptyText?: string;
}

export function Table<T extends Record<string, any>>({
  columns,
  data,
  keyField,
  onRowClick,
  selectedKey,
  className = '',
  emptyText = 'No records found',
}: TableProps<T>) {
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const handleSort = (colKey: string) => {
    if (sortCol === colKey) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortCol(colKey);
      setSortDir('asc');
    }
  };

  const sortedData = React.useMemo(() => {
    if (!sortCol) return data;
    return [...data].sort((a, b) => {
      const aVal = a[sortCol];
      const bVal = b[sortCol];
      if (aVal === bVal) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;
      const res = aVal > bVal ? 1 : -1;
      return sortDir === 'asc' ? res : -res;
    });
  }, [data, sortCol, sortDir]);

  const getKey = (row: T) => {
    if (typeof keyField === 'function') return keyField(row);
    return String(row[keyField]);
  };

  return (
    <div className={`overflow-x-auto border border-[var(--c-border-base)] rounded-lg bg-[var(--c-bg-surface)] shadow-sm ${className}`}>
      <table className="w-full text-left border-collapse text-xs font-ui">
        <thead className="bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] sticky top-0 z-10 select-none">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                style={{ width: col.width }}
                onClick={() => col.sortable && handleSort(col.key)}
                className={`py-2.5 px-3.5 text-xs text-[var(--c-text-mid)] font-medium ${col.sortable ? 'cursor-pointer hover:text-[var(--c-text-high)]' : ''
                  } ${col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'}`}
              >
                <div className={`inline-flex items-center gap-1.5 ${col.align === 'right' ? 'justify-end' : ''}`}>
                  <span>{col.header}</span>
                  {col.sortable && sortCol === col.key && (
                    <span className="text-[var(--c-accent)] text-[10px]">
                      {sortDir === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--c-border-base)]">
          {sortedData.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                className="py-8 text-center text-[var(--c-text-low)] font-ui text-xs"
              >
                {emptyText}
              </td>
            </tr>
          ) : (
            sortedData.map((row, idx) => {
              const k = getKey(row);
              const isSelected = selectedKey === k;
              return (
                <tr
                  key={k}
                  onClick={() => onRowClick && onRowClick(row)}
                  className={`transition-colors select-none ${onRowClick ? 'cursor-pointer' : ''
                    } ${isSelected
                      ? 'bg-[var(--c-bg-raised)] text-[var(--c-text-max)]'
                      : 'hover:bg-[var(--c-bg-raised)]/50 text-[var(--c-text-high)]'
                    }`}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={`py-2.5 px-3.5 whitespace-nowrap ${col.align === 'right'
                          ? 'text-right tabular-nums'
                          : col.align === 'center'
                            ? 'text-center'
                            : 'text-left'
                        }`}
                    >
                      {col.render ? col.render(row, idx) : row[col.key]}
                    </td>
                  ))}
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
