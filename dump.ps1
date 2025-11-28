Get-ChildItem -Path .\output\image_buffers -Filter *.bin -File | ForEach-Object {
    $log = Join-Path $_.DirectoryName ($_.BaseName + '.log')

    # ファイル名を先頭に書き込み
    "File: $($_.FullName)`r`n" | Out-File -FilePath $log -Encoding utf8

    # パケット/フレーム情報 + VPS/SPS/PPS (extradata) を JSON で出力
    & ffprobe -v error -f hevc `
        -show_packets -show_frames `
        -show_entries stream=codec_type,codec_name,extradata `
        -select_streams v:0 `
        -print_format json `
        $_.FullName 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
}
