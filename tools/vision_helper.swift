// vision_helper — 미리보기 이미지에 공개 Vision 분류기(VNClassifyImageRequest)와 한국어 OCR(VNRecognizeTextRequest)을
// 돌려 결과를 JSON 한 줄씩 stdout으로 낸다. 네트워크 0, 이미지 파일은 읽기만 한다.
//
// 사용: vision_helper <classify|ocr> <목록 파일: 한 줄에 "uuid<TAB>이미지 경로">
// 환경변수: MIN_CONF  분류 라벨 하한(기본 0.02 — 이 값 미만 라벨은 내지 않는다)
//          OCR_LANGS  OCR 언어(기본 ko-KR,en-US — 한국어는 accurate 모드만 지원)
//          MAXPX      긴 변 상한 픽셀(기본 0 = 원본 크기 그대로, 키우지 않는다)
// 출력 줄: {"meta":{...}} 1줄 → 이미지마다 {"uuid":..,"w":..,"h":..,"ms":..,"labels":[[id,conf],..]} 또는
//          {"uuid":..,"w":..,"h":..,"ms":..,"lines":[[글자,conf],..]} 또는 {"uuid":..,"error":"..."}
// 종료 코드: 0 = 목록 끝까지 처리(이미지별 실패는 error 줄로 냄), 2 = 사용법 오류
import Foundation
import ImageIO
import Vision

setvbuf(stdout, nil, _IOLBF, 0)
let env = ProcessInfo.processInfo.environment
let minConf = Float(env["MIN_CONF"] ?? "0.02") ?? 0.02
let ocrLangs = (env["OCR_LANGS"] ?? "ko-KR,en-US").split(separator: ",").map(String.init)
let maxPx = Int(env["MAXPX"] ?? "0") ?? 0

func emit(_ obj: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
          let line = String(data: data, encoding: .utf8) else {
        print("{\"error\":\"json-encode\"}")
        return
    }
    print(line)
}

func errLine(_ s: String) {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

func loadImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    var longSide = maxPx
    if longSide <= 0 {
        guard let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
        longSide = max(w, h)
    }
    // 썸네일 경로를 쓰는 이유: EXIF 회전(kCGImageSourceCreateThumbnailWithTransform)을 적용한 비트맵을 얻기 위해서.
    let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                 kCGImageSourceThumbnailMaxPixelSize: longSide,
                                 kCGImageSourceCreateThumbnailWithTransform: true]
    return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
}

func r4(_ x: Float) -> Double { (Double(x) * 10000).rounded() / 10000 }

let args = CommandLine.arguments
guard args.count == 3, ["classify", "ocr"].contains(args[1]),
      let list = try? String(contentsOfFile: args[2], encoding: .utf8) else {
    errLine("usage: vision_helper <classify|ocr> <listfile: uuid<TAB>path per line>")
    exit(2)
}
let mode = args[1]
let items: [(String, String)] = list.split(separator: "\n").compactMap { line in
    let parts = line.split(separator: "\t", maxSplits: 1).map(String.init)
    return parts.count == 2 ? (parts[0], parts[1]) : nil
}

let classifyProbe = VNClassifyImageRequest()
let ocrProbe = VNRecognizeTextRequest()
emit(["meta": ["mode": mode, "items": items.count, "min_conf": r4(minConf), "ocr_langs": ocrLangs,
               "classify_revision": classifyProbe.revision, "ocr_revision": ocrProbe.revision,
               "os": ProcessInfo.processInfo.operatingSystemVersionString]])

var ok = 0, fail = 0
for (uuid, path) in items {
    guard let img = loadImage(path) else {
        fail += 1
        emit(["uuid": uuid, "error": "LOADFAIL"])
        continue
    }
    let handler = VNImageRequestHandler(cgImage: img, options: [:])
    let t0 = DispatchTime.now().uptimeNanoseconds
    do {
        var out: [String: Any] = ["uuid": uuid, "w": img.width, "h": img.height]
        if mode == "classify" {
            let req = VNClassifyImageRequest()
            try handler.perform([req])
            let obs = (req.results ?? []).filter { $0.confidence >= minConf }.sorted { $0.confidence > $1.confidence }
            out["labels"] = obs.map { [$0.identifier, r4($0.confidence)] as [Any] }
        } else {
            let req = VNRecognizeTextRequest()
            req.recognitionLevel = .accurate
            req.recognitionLanguages = ocrLangs
            req.usesLanguageCorrection = true
            try handler.perform([req])
            let lines = (req.results ?? []).compactMap { $0.topCandidates(1).first }
            out["lines"] = lines.map { [$0.string, r4($0.confidence)] as [Any] }
        }
        out["ms"] = (Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e5).rounded() / 10
        emit(out)
        ok += 1
    } catch {
        fail += 1
        emit(["uuid": uuid, "error": "VNFAIL \(error)"])
    }
}
errLine("done ok=\(ok) fail=\(fail)")
