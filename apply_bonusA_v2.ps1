$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$HelperPath = Join-Path $ProjectRoot "src\evaluation\faithful_explanation.py"
$AppPath = Join-Path $ProjectRoot "app.py"
$SourceHelper = Join-Path $ProjectRoot "faithful_explanation_patch_source.py"

if (-not (Test-Path $AppPath)) {
    throw "app.py not found in $ProjectRoot"
}

if (-not (Test-Path $SourceHelper)) {
    throw "faithful_explanation_patch_source.py not found in $ProjectRoot"
}

New-Item -ItemType Directory -Force -Path (Split-Path $HelperPath) | Out-Null
Copy-Item -Force $SourceHelper $HelperPath

$app = Get-Content -Raw -Path $AppPath

# Add helper imports once.
if ($app -notmatch 'faithful_explanation') {
    $anchor = 'from src.evaluation.metadata import extract_exif'
    if ($app -notmatch [regex]::Escape($anchor)) {
        throw "Could not find the metadata import anchor in app.py."
    }

    $imports = "from src.evaluation.metadata import extract_exif`r`nfrom src.evaluation.faithful_explanation import (`r`n    analyze_localization,`r`n    build_grounded_evidence,`r`n    draw_evidence_region,`r`n    make_heatmap_image,`r`n)"
    $app = $app.Replace($anchor, $imports)
}

# Do not try to replace the older Explanation block.
# Different app.py revisions use slightly different indentation/comments.
# Inject Bonus A immediately before the existing Image Metadata section.
if ($app -notmatch 'analyze_localization\(result\["heatmap"\]\)') {

    $markerPattern = '(?ms)^\s*st\.markdown\(\s*''<div class="section-title">Image Metadata</div>'',\s*unsafe_allow_html=True,\s*\)'

    $bonusBlock = @'
    # -----------------------------------------------------
    # Bonus A — Faithful Explanation / Localization
    # -----------------------------------------------------
    localization = analyze_localization(result["heatmap"])
    explanation = build_grounded_evidence(
        ai_probability=ai_probability,
        predicted_class=predicted_class,
        localization=localization,
    )

    st.divider()
    st.markdown(
        '<div class="section-title">Faithful Explanation</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**MODEL EVIDENCE**")
    for item in explanation["evidence"]:
        st.write(f"• {item}")

    st.markdown("**UNCERTAINTY**")
    st.info(explanation["uncertainty"])

    original_image = image.copy()
    heatmap_image = make_heatmap_image(
        result["heatmap"],
        size=original_image.size,
    )
    overlay_image = make_overlay(
        original_image,
        result["heatmap"],
    )
    localized_image = draw_evidence_region(
        original_image,
        explanation["bbox"],
    )

    visual_col1, visual_col2, visual_col3 = st.columns(3)

    with visual_col1:
        st.image(
            original_image,
            caption="Original image",
            width="stretch",
        )

    with visual_col2:
        st.image(
            heatmap_image,
            caption="Artifact heatmap",
            width="stretch",
        )

    with visual_col3:
        overlay_caption = (
            "Model evidence overlay — localized"
            if explanation["bbox"] is not None
            else "Model evidence overlay — diffuse evidence"
        )
        st.image(
            overlay_image,
            caption=overlay_caption,
            width="stretch",
        )

    if explanation["bbox"] is not None:
        st.image(
            localized_image,
            caption="Primary evidence region",
            width="stretch",
        )
        st.caption(
            "The region box is shown only when the measured activation is "
            "genuinely localized. It is not forced around diffuse evidence."
        )
    else:
        st.info(
            "The model evidence is distributed across the image rather than "
            "concentrated in a single region, so no primary evidence box is shown."
        )

    st.markdown(
        f"""
        **Localization diagnostics**

        - Heatmap mean: `{localization.mean_activation:.3f}`
        - Maximum activation: `{localization.max_activation:.3f}`
        - High-activation area: `{localization.active_fraction:.1%}`
        - Top-10% / mean concentration: `{localization.concentration_ratio:.2f}×`
        - Decision explained: `{"AI-generated" if predicted_class == 1 else "REAL"}`
        """
    )

    st.caption(
        "The explanation is derived deterministically from the existing "
        "trained model's Grad-CAM output and final prediction. It does not "
        "invent semantic artifacts and does not use an external LLM or API."
    )

'@

    if (-not [regex]::IsMatch($app, $markerPattern)) {
        throw "Could not find the Image Metadata section in app.py. No changes were made to app.py."
    }

    $replacement = $bonusBlock + "`r`n    st.markdown('<div class=""section-title"">Image Metadata</div>', unsafe_allow_html=True)"
    $app = [regex]::Replace($app, $markerPattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $replacement }, 1)
}

Set-Content -Path $AppPath -Value $app -Encoding UTF8

python -m py_compile $HelperPath $AppPath

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "SignalScope Bonus A patch applied successfully." -ForegroundColor Green
Write-Host "Python syntax check PASSED." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Run:"
Write-Host "  streamlit run app.py"
