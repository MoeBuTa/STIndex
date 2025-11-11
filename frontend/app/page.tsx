'use client'

import { useEffect, useState } from 'react'
import { DashboardStats } from './components/DashboardStats'
import { TemporalTimeline } from './components/TemporalTimeline'
import { DimensionBreakdown } from './components/DimensionBreakdown'

interface ExtractionResult {
  chunk_id: string
  chunk_index: number
  document_id: string
  document_title: string
  extraction: {
    entities?: {
      temporal?: any[]
      spatial?: any[]
      [key: string]: any[] | undefined
    }
    temporal_entities: any[]
    spatial_entities: any[]
    event_type?: any[]
    disease?: any[]
    venue_type?: any[]
    success: boolean
    error?: string
    document_metadata: {
      source: string
      category: string
      topic: string
      jurisdiction: string
      year: number
    }
    extraction_config?: {
      enabled_dimensions?: string[]
      dimension_config_path?: string
    }
    dimension_configs?: Record<string, any>
  }
}

export default function Home() {
  const [data, setData] = useState<ExtractionResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/data/extraction_results.json')
      .then((res) => res.json())
      .then((results) => {
        setData(results)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600">Loading extraction data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-red-500 text-xl">Error loading data: {error}</p>
        </div>
      </div>
    )
  }

  // Filter successful extractions
  const successfulExtractions = data.filter((item) => item.extraction.success)

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-7xl">
        <div className="space-y-8">
          <div>
            <h1 className="text-4xl font-bold mb-2">STIndex Dashboard</h1>
            <p className="text-gray-600 text-lg">
              Multi-Dimensional Data Visualization
            </p>
          </div>

          <DashboardStats data={successfulExtractions} />

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Temporal Timeline</h2>
            <TemporalTimeline data={successfulExtractions} />
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Dimension Analysis</h2>
            <DimensionBreakdown data={successfulExtractions} />
          </div>
        </div>
      </div>
    </div>
  )
}
