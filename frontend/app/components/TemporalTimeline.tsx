'use client'

import { useMemo } from 'react'

interface TemporalEntity {
  text: string
  normalized?: string
  confidence?: number
  normalization_type?: string
  dimension_name?: string
  reflection_scores?: {
    relevance: number
    accuracy: number
    completeness: number
    consistency: number
    reasoning: string
  }
}

interface ExtractionResult {
  chunk_id: string
  document_id: string
  document_title: string
  extraction: {
    temporal_entities: TemporalEntity[]
    document_metadata: {
      source: string
      category: string
      year: number
    }
  }
}

interface TemporalTimelineProps {
  data: ExtractionResult[]
}

export function TemporalTimeline({ data }: TemporalTimelineProps) {
  // Extract all temporal entities and group by year
  const temporalData = useMemo(() => {
    const entities: Array<TemporalEntity & { document_title: string; source: string }> = []

    data.forEach((item) => {
      item.extraction.temporal_entities.forEach((entity) => {
        entities.push({
          ...entity,
          document_title: item.document_title,
          source: item.extraction.document_metadata.source,
        })
      })
    })

    // Sort by normalized date (if available)
    entities.sort((a, b) => {
      const dateA = a.normalized || ''
      const dateB = b.normalized || ''
      return dateA.localeCompare(dateB)
    })

    return entities
  }, [data])

  if (temporalData.length === 0) {
    return (
      <div className="p-4 bg-gray-50 rounded-md">
        <p className="text-gray-500">No temporal entities found</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {temporalData.map((entity, index) => (
        <div
          key={index}
          className="p-4 bg-gray-50 border-l-4 border-blue-500 rounded-md"
        >
          <div className="flex justify-between items-start">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-bold">{entity.text}</span>
                {entity.normalization_type && (
                  <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs rounded">
                    {entity.normalization_type}
                  </span>
                )}
                {entity.confidence && (
                  <span className="px-2 py-0.5 bg-green-100 text-green-800 text-xs rounded">
                    {Math.round(entity.confidence * 100)}%
                  </span>
                )}
              </div>
              {entity.normalized && (
                <p className="text-sm text-gray-600 mb-1">
                  Normalized: {entity.normalized}
                </p>
              )}
              <p className="text-xs text-gray-500">
                Source: {entity.source} • {entity.document_title}
              </p>
            </div>

            {entity.reflection_scores && (
              <div className="text-right">
                <p className="text-xs text-gray-600 mb-1">Quality Scores:</p>
                <p className="text-xs">R: {entity.reflection_scores.relevance.toFixed(2)}</p>
                <p className="text-xs">A: {entity.reflection_scores.accuracy.toFixed(2)}</p>
                <p className="text-xs">C: {entity.reflection_scores.consistency.toFixed(2)}</p>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
