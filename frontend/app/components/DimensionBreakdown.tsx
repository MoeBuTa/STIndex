'use client'

import { useMemo, useState } from 'react'

interface DimensionEntity {
  text: string
  category?: string
  confidence?: number
  dimension_name?: string
  [key: string]: any
}

interface ExtractionResult {
  chunk_id: string
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
    document_metadata: {
      source: string
      category: string
    }
  }
}

interface DimensionBreakdownProps {
  data: ExtractionResult[]
}

export function DimensionBreakdown({ data }: DimensionBreakdownProps) {
  const [activeTab, setActiveTab] = useState<string>('')

  // Extract all custom dimensions (not temporal/spatial)
  const dimensions = useMemo(() => {
    const dimensionMap = new Map<string, DimensionEntity[]>()

    data.forEach((item) => {
      // First, check extraction.entities object
      if (item.extraction.entities) {
        Object.entries(item.extraction.entities).forEach(([dimName, entities]) => {
          // Skip temporal and spatial as they're handled separately
          if (dimName === 'temporal' || dimName === 'spatial' || !entities) {
            return
          }
          if (!dimensionMap.has(dimName)) {
            dimensionMap.set(dimName, [])
          }
          entities.forEach((entity) => {
            dimensionMap.get(dimName)?.push({
              ...entity,
              source: item.extraction.document_metadata.source,
              document_title: item.document_title,
            } as DimensionEntity)
          })
        })
      }

      // Also check for top-level dimension arrays (event_type, disease, venue_type, etc.)
      const customDimensions = ['event_type', 'disease', 'venue_type']
      customDimensions.forEach((dimName) => {
        const entities = (item.extraction as any)[dimName]
        if (entities && Array.isArray(entities) && entities.length > 0) {
          if (!dimensionMap.has(dimName)) {
            dimensionMap.set(dimName, [])
          }
          entities.forEach((entity) => {
            // Check if this entity already exists (to avoid duplicates from entities object)
            const existing = dimensionMap.get(dimName)?.find(
              (e) => e.text === entity.text && (e as any).source === item.extraction.document_metadata.source
            )
            if (!existing) {
              dimensionMap.get(dimName)?.push({
                ...entity,
                source: item.extraction.document_metadata.source,
                document_title: item.document_title,
              } as DimensionEntity)
            }
          })
        }
      })
    })

    return dimensionMap
  }, [data])

  // Set initial active tab
  useMemo(() => {
    if (dimensions.size > 0 && !activeTab) {
      setActiveTab(Array.from(dimensions.keys())[0])
    }
  }, [dimensions, activeTab])

  // Calculate category frequencies for a dimension
  const getCategoryStats = (entities: DimensionEntity[]) => {
    const categoryCount = new Map<string, number>()
    entities.forEach((entity) => {
      const category = entity.category || entity.text || 'unknown'
      categoryCount.set(category, (categoryCount.get(category) || 0) + 1)
    })

    const total = entities.length
    return Array.from(categoryCount.entries())
      .map(([category, count]) => ({
        category,
        count,
        percentage: (count / total) * 100,
      }))
      .sort((a, b) => b.count - a.count)
  }

  if (dimensions.size === 0) {
    return (
      <div className="p-4 bg-gray-50 rounded-md">
        <p className="text-gray-500">No custom dimensions found</p>
      </div>
    )
  }

  const formatDimensionName = (name: string) => {
    return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
  }

  return (
    <div>
      {/* Tab navigation */}
      <div className="flex flex-wrap gap-2 mb-6 border-b border-gray-200">
        {Array.from(dimensions.keys()).map((dimName) => (
          <button
            key={dimName}
            onClick={() => setActiveTab(dimName)}
            className={`px-4 py-2 font-medium text-sm transition-colors ${
              activeTab === dimName
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {formatDimensionName(dimName)}
            <span className="ml-2 px-2 py-0.5 bg-gray-200 text-gray-700 text-xs rounded">
              {dimensions.get(dimName)?.length || 0}
            </span>
          </button>
        ))}
      </div>

      {/* Tab content */}
      {Array.from(dimensions.entries()).map(([dimName, entities]) => {
        if (dimName !== activeTab) return null

        const categoryStats = getCategoryStats(entities)

        return (
          <div key={dimName} className="space-y-6">
            {/* Category distribution */}
            <div>
              <h3 className="text-lg font-bold mb-3">Category Distribution</h3>
              <div className="space-y-2">
                {categoryStats.map((stat) => (
                  <div key={stat.category}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">{stat.category}</span>
                      <div className="flex gap-2">
                        <span className="text-sm text-gray-600">{stat.count}</span>
                        <span className="text-sm text-gray-500">
                          ({stat.percentage.toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                    <div className="w-full h-2 bg-gray-200 rounded-md overflow-hidden">
                      <div
                        className="h-full bg-blue-500 transition-all"
                        style={{ width: `${stat.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Entity list */}
            <div>
              <h3 className="text-lg font-bold mb-3">All Entities</h3>
              <div className="space-y-2">
                {entities.map((entity, index) => (
                  <div
                    key={index}
                    className="p-3 bg-gray-50 rounded-md border-l-3 border-blue-400"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{entity.text}</span>
                          {entity.category && (
                            <span className="px-2 py-0.5 bg-purple-100 text-purple-800 text-xs rounded">
                              {entity.category}
                            </span>
                          )}
                          {entity.confidence && (
                            <span className="px-2 py-0.5 bg-green-100 text-green-800 text-xs rounded">
                              {Math.round(entity.confidence * 100)}%
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-gray-500">
                          {(entity as any).source} • {(entity as any).document_title}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
