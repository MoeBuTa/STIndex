/**
 * Client-side analytics for spatiotemporal event clustering and story detection
 * Implements DBSCAN-inspired clustering for spatiotemporal events
 */

export interface SpatioTemporalEvent {
  id: string
  text: string
  latitude?: number
  longitude?: number
  timestamp?: string
  normalized_date?: string
  location?: string
  normalized_location?: string
  category?: string
  confidence?: number
  document_id: string
  document_title: string
  chunk_id?: string
  source: string
  custom_dimensions?: Record<string, any>
  reflection_scores?: {
    relevance: number
    accuracy: number
    completeness: number
    consistency: number
    reasoning?: string
  }
}

export interface EventCluster {
  id: string
  events: SpatioTemporalEvent[]
  centroid: { lat: number; lng: number }
  timeRange: { start: string; end: string }
  dominantCategory?: string
  size: number
  density: number
  entities: Set<string>
}

export interface BurstPeriod {
  start: string
  end: string
  peakTime: string
  eventCount: number
  intensity: number
  dominantLocation?: string
  dominantCategory?: string
}

/**
 * Calculate Haversine distance between two coordinates (in km)
 */
export function haversineDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371 // Earth's radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lon2 - lon1) * Math.PI) / 180
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return R * c
}

/**
 * Calculate temporal distance in days
 */
export function temporalDistance(date1: string, date2: string): number {
  const d1 = new Date(date1)
  const d2 = new Date(date2)
  return Math.abs((d2.getTime() - d1.getTime()) / (1000 * 60 * 60 * 24))
}

/**
 * DBSCAN-inspired spatiotemporal clustering
 * @param events - Array of spatiotemporal events
 * @param spatialEps - Spatial radius in km
 * @param temporalEps - Temporal window in days
 * @param minPoints - Minimum points to form a cluster
 */
export function clusterEvents(
  events: SpatioTemporalEvent[],
  spatialEps: number = 50,
  temporalEps: number = 7,
  minPoints: number = 3
): EventCluster[] {
  const clusters: EventCluster[] = []
  const visited = new Set<number>()
  const clustered = new Set<number>()

  // Filter events with valid coordinates and timestamps
  const validEvents = events.filter(
    (e) => e.latitude && e.longitude && (e.timestamp || e.normalized_date)
  )

  for (let i = 0; i < validEvents.length; i++) {
    if (visited.has(i)) continue
    visited.add(i)

    const neighbors = getNeighbors(validEvents, i, spatialEps, temporalEps)

    if (neighbors.length >= minPoints) {
      const clusterEvents: SpatioTemporalEvent[] = []
      const queue = [...neighbors]

      while (queue.length > 0) {
        const idx = queue.shift()!
        if (!clustered.has(idx)) {
          clustered.add(idx)
          clusterEvents.push(validEvents[idx])

          if (!visited.has(idx)) {
            visited.add(idx)
            const newNeighbors = getNeighbors(validEvents, idx, spatialEps, temporalEps)
            if (newNeighbors.length >= minPoints) {
              queue.push(...newNeighbors)
            }
          }
        }
      }

      if (clusterEvents.length > 0) {
        clusters.push(createCluster(clusterEvents, clusters.length))
      }
    }
  }

  return clusters.sort((a, b) => b.size - a.size)
}

function getNeighbors(
  events: SpatioTemporalEvent[],
  idx: number,
  spatialEps: number,
  temporalEps: number
): number[] {
  const neighbors: number[] = []
  const event = events[idx]

  for (let i = 0; i < events.length; i++) {
    if (i === idx) continue

    const other = events[i]
    const spatialDist = haversineDistance(
      event.latitude!,
      event.longitude!,
      other.latitude!,
      other.longitude!
    )
    const temporalDist = temporalDistance(
      event.timestamp || event.normalized_date!,
      other.timestamp || other.normalized_date!
    )

    if (spatialDist <= spatialEps && temporalDist <= temporalEps) {
      neighbors.push(i)
    }
  }

  return neighbors
}

function createCluster(events: SpatioTemporalEvent[], id: number): EventCluster {
  const lats = events.map((e) => e.latitude!)
  const lngs = events.map((e) => e.longitude!)
  const timestamps = events.map((e) => e.timestamp || e.normalized_date!)

  const centroid = {
    lat: lats.reduce((a, b) => a + b, 0) / lats.length,
    lng: lngs.reduce((a, b) => a + b, 0) / lngs.length,
  }

  const sortedTimes = timestamps.sort()
  const timeRange = {
    start: sortedTimes[0],
    end: sortedTimes[sortedTimes.length - 1],
  }

  // Find dominant category
  const categoryCount = new Map<string, number>()
  events.forEach((e) => {
    if (e.category) {
      categoryCount.set(e.category, (categoryCount.get(e.category) || 0) + 1)
    }
  })
  const dominantCategory = Array.from(categoryCount.entries()).sort(
    (a, b) => b[1] - a[1]
  )[0]?.[0]

  // Collect unique entities
  const entities = new Set<string>()
  events.forEach((e) => {
    entities.add(e.text)
    if (e.custom_dimensions) {
      Object.values(e.custom_dimensions).forEach((dim) => {
        if (Array.isArray(dim)) {
          dim.forEach((item: any) => {
            if (typeof item === 'object' && item.text) {
              entities.add(item.text)
            }
          })
        }
      })
    }
  })

  // Calculate density (events per km² per day)
  const maxDistance = Math.max(
    ...events.flatMap((e1) =>
      events.map((e2) =>
        haversineDistance(e1.latitude!, e1.longitude!, e2.latitude!, e2.longitude!)
      )
    )
  )
  const timeSpan = temporalDistance(timeRange.start, timeRange.end) || 1
  const density = events.length / (Math.PI * maxDistance * maxDistance * timeSpan)

  return {
    id: `cluster-${id}`,
    events,
    centroid,
    timeRange,
    dominantCategory,
    size: events.length,
    density,
    entities,
  }
}

/**
 * Detect bursts of activity (temporal spikes)
 * @param events - Array of spatiotemporal events
 * @param windowSize - Time window in days
 * @param threshold - Minimum events to qualify as burst
 */
export function detectBursts(
  events: SpatioTemporalEvent[],
  windowSize: number = 1,
  threshold: number = 5
): BurstPeriod[] {
  // Group events by date
  const dateGroups = new Map<string, SpatioTemporalEvent[]>()

  events.forEach((event) => {
    const date = event.timestamp || event.normalized_date
    if (!date) return

    const dateKey = date.split('T')[0]
    if (!dateGroups.has(dateKey)) {
      dateGroups.set(dateKey, [])
    }
    dateGroups.get(dateKey)!.push(event)
  })

  const sortedDates = Array.from(dateGroups.keys()).sort()
  const bursts: BurstPeriod[] = []

  for (let i = 0; i < sortedDates.length; i++) {
    const windowEnd = i + Math.floor(windowSize)
    if (windowEnd >= sortedDates.length) break

    const windowEvents = sortedDates
      .slice(i, windowEnd + 1)
      .flatMap((date) => dateGroups.get(date) || [])

    if (windowEvents.length >= threshold) {
      // Find peak day
      const dayCounts = new Map<string, number>()
      sortedDates.slice(i, windowEnd + 1).forEach((date) => {
        dayCounts.set(date, dateGroups.get(date)?.length || 0)
      })
      const peakDay = Array.from(dayCounts.entries()).sort((a, b) => b[1] - a[1])[0]

      // Find dominant location and category
      const locationCount = new Map<string, number>()
      const categoryCount = new Map<string, number>()

      windowEvents.forEach((e) => {
        if (e.text) {
          locationCount.set(e.text, (locationCount.get(e.text) || 0) + 1)
        }
        if (e.category) {
          categoryCount.set(e.category, (categoryCount.get(e.category) || 0) + 1)
        }
      })

      const dominantLocation = Array.from(locationCount.entries()).sort(
        (a, b) => b[1] - a[1]
      )[0]?.[0]

      const dominantCategory = Array.from(categoryCount.entries()).sort(
        (a, b) => b[1] - a[1]
      )[0]?.[0]

      bursts.push({
        start: sortedDates[i],
        end: sortedDates[windowEnd],
        peakTime: peakDay[0],
        eventCount: windowEvents.length,
        intensity: windowEvents.length / (windowEnd - i + 1),
        dominantLocation,
        dominantCategory,
      })
    }
  }

  return bursts.sort((a, b) => b.intensity - a.intensity)
}

/**
 * Calculate entity co-occurrence network
 */
export function calculateCoOccurrence(
  events: SpatioTemporalEvent[]
): Map<string, Map<string, number>> {
  const coOccurrence = new Map<string, Map<string, number>>()

  // Group events by document
  const docGroups = new Map<string, SpatioTemporalEvent[]>()
  events.forEach((event) => {
    if (!docGroups.has(event.document_id)) {
      docGroups.set(event.document_id, [])
    }
    docGroups.get(event.document_id)!.push(event)
  })

  // Calculate co-occurrence within documents
  docGroups.forEach((docEvents) => {
    const entities = docEvents.map((e) => e.text)

    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const e1 = entities[i]
        const e2 = entities[j]

        if (!coOccurrence.has(e1)) {
          coOccurrence.set(e1, new Map())
        }
        if (!coOccurrence.has(e2)) {
          coOccurrence.set(e2, new Map())
        }

        coOccurrence.get(e1)!.set(e2, (coOccurrence.get(e1)!.get(e2) || 0) + 1)
        coOccurrence.get(e2)!.set(e1, (coOccurrence.get(e2)!.get(e1) || 0) + 1)
      }
    }
  })

  return coOccurrence
}
