import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'STIndex - Multi-Dimensional Data Visualization',
  description: 'Interactive dashboard for spatiotemporal and dimensional data exploration',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
