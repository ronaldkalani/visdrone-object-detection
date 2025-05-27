from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize once
tracker = DeepSort(max_age=30)

def track_objects(detections, image):
    # detections: list of [x1, y1, x2, y2, confidence, class]
    tracks = tracker.update_tracks(detections, frame=image)

    tracked = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        tracked.append({
            'track_id': track.track_id,
            'bbox': ltrb
        })
    return tracked
