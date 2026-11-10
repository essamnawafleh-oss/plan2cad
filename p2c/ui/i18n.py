"""Arabic / English strings."""
LANG = {'cur': 'ar'}

S = {
    'app': ('Plan2CAD — تحويل المخططات إلى أوتوكاد — عصام النوافلة', 'Plan2CAD — Scanned plans to CAD — Essam Nawafleh'),
    'open_image': ('فتح صورة', 'Open image'),
    'open_project': ('فتح مشروع', 'Open project'),
    'save': ('حفظ', 'Save'),
    'save_as': ('حفظ باسم', 'Save as'),
    'export': ('تصدير DXF', 'Export DXF'),
    'auto': ('كشف أوتوماتيكي', 'Auto detect'),
    'undo': ('تراجع', 'Undo'), 'redo': ('إعادة', 'Redo'),
    'fit': ('ملاءمة العرض', 'Fit view'),
    'lang_btn': ('English', 'عربي'),
    'file': ('ملف', 'File'), 'edit': ('تحرير', 'Edit'), 'view': ('عرض', 'View'), 'tools': ('أدوات', 'Tools'),
    'help': ('مساعدة', 'Help'), 'shortcuts': ('اختصارات لوحة المفاتيح', 'Keyboard shortcuts'),
    'about': ('حول البرنامج', 'About'),
    'delete': ('حذف المحدد', 'Delete selected'),
    'select_all': ('تحديد الكل', 'Select all'),
    'clear_layer': ('مسح طبقة…', 'Clear layer…'),
    # tools
    't_select': ('تحديد / تحريك', 'Select / move'),
    't_wall': ('جدار', 'Wall'), 't_door': ('باب', 'Door'), 't_door2': ('باب مزدوج', 'Double door'),
    't_window': ('شباك', 'Window'), 't_column': ('عمود', 'Column'), 't_stair': ('درج', 'Stair'),
    't_poly': ('خط حر', 'Polyline'), 't_text': ('نص', 'Text'), 't_level': ('منسوب', 'Level'),
    't_calib': ('معايرة المقياس', 'Calibrate scale'), 't_furn': ('وضع قطعة', 'Place block'),
    # hints
    'h_select': ('اضغط لتحديد عنصر، واسحب لتحريكه. Del للحذف — R تدوير 90° — Q/E تدوير 15° — M عكس — F قلب فتحة الباب.',
                 'Click to select, drag to move. Del delete — R rotate 90° — Q/E rotate 15° — M mirror — F flip door swing.'),
    'h_wall': ('اضغط لتحديد نقاط الجدار. Enter أو دبل كليك للإنهاء — Esc للإلغاء — Shift لإلغاء الالتقاط.',
               'Click wall points. Enter / double-click to finish — Esc cancel — hold Shift to disable snapping.'),
    'h_door': ('اضغط على طرف المفصّلة، ثم على الطرف الثاني، ثم على جهة فتح الباب.',
               'Click the hinge jamb, then the other jamb, then the side the door opens to.'),
    'h_door2': ('اضغط على طرفي الفتحة، ثم على جهة فتح البابين.', 'Click both jambs, then the side the leaves open to.'),
    'h_window': ('اضغط على بداية الشباك ثم نهايته فوق الجدار.', 'Click the start and end of the window on the wall.'),
    'h_column': ('اضغط لوضع عمود بالمقاس المحدد.', 'Click to place a column of the current size.'),
    'h_stair': ('اضغط على بداية الدرج، ثم نهايته (اتجاه الطلوع)، ثم على عرضه.',
                'Click stair start, then end (up direction), then its width.'),
    'h_poly': ('اضغط لرسم خط حر على الطبقة المختارة. Enter للإنهاء — C لإغلاقه.',
               'Click to draw a polyline on the chosen layer. Enter to finish — C to close.'),
    'h_text': ('اضغط لوضع نص.', 'Click to place a text.'),
    'h_level': ('اضغط لوضع رمز منسوب.', 'Click to place a level mark.'),
    'h_calib': ('اضغط على نقطتين بتعرف المسافة الحقيقية بينهم (مثلاً عرض باب أو بُعد مكتوب).',
                'Click two points whose real distance you know (a door width or a written dimension).'),
    'h_furn': ('اضغط على القطعة بالسكان (بتنقاس تلقائياً) أو بأي مكان. R تدوير 90° — Q/E تدوير 15° — M عكس — Esc للخروج.',
               'Click the piece in the scan (auto-measured) or anywhere. R rotate 90° — Q/E rotate 15° — M mirror — Esc to stop.'),
    'fit_mode': ('ملاءمة القطعة للرسم', 'Fit block to drawing'),
    'fit_tip': ('لما تضغط على قطعة مرسومة بالسكان، البرنامج بيقيس شكلها وبيحط البلوك بنفس المقاس والاتجاه.',
                'When you click a drawn piece in the scan, its shape is measured and the block is placed at the same size and angle.'),
    'h_start': ('ابدأ بفتح صورة المخطط (ملف ← فتح صورة).', 'Start by opening a plan image (File → Open image).'),
    # panels
    'layers': ('الطبقات', 'Layers'), 'library': ('مكتبة البلوكات', 'Block library'),
    'properties': ('الخصائص', 'Properties'), 'nothing_sel': ('ما في عنصر محدد', 'Nothing selected'),
    'search': ('بحث…', 'Search…'),
    'thickness': ('السماكة (م)', 'Thickness (m)'), 'width': ('العرض (م)', 'Width (m)'),
    'depth': ('العمق (م)', 'Depth (m)'), 'rotation': ('الدوران (°)', 'Rotation (°)'),
    'mirror': ('معكوس', 'Mirrored'), 'block': ('البلوك', 'Block'), 'layer': ('الطبقة', 'Layer'),
    'size': ('المقاس (م)', 'Size (m)'), 'double': ('باب مزدوج', 'Double door'),
    'flip': ('قلب جهة الفتح', 'Flip swing'), 'hinge': ('تبديل المفصّلة', 'Swap hinge'),
    'tread': ('عمق الدرجة (م)', 'Tread (m)'), 'text': ('النص', 'Text'), 'height': ('الارتفاع (م)', 'Height (m)'),
    'value': ('القيمة', 'Value'), 'closed': ('مغلق', 'Closed'), 'n_selected': ('العناصر المحددة', 'Selected'),
    'cur_thick': ('سماكة الجدار', 'Wall thickness'), 'cur_layer': ('طبقة الخط', 'Polyline layer'),
    'apply_all': ('تطبيق على المحدد', 'Apply to selection'),
    # dialogs
    'import_title': ('تجهيز الصورة', 'Prepare image'),
    'rotate': ('تدوير الصورة', 'Rotate image'), 'deskew': ('تصحيح الميلان تلقائياً', 'Auto-straighten'),
    'clean_bg': ('تنظيف خلفية السكان', 'Clean scan background'),
    'scale_m_px': ('المقياس (متر لكل بكسل)', 'Scale (metres per pixel)'),
    'scale_note': ('إذا ما بتعرف المقياس اتركه، وبعدين استعمل أداة "معايرة المقياس".',
                   'Unknown? Leave it and use the "Calibrate scale" tool afterwards.'),
    'ok': ('موافق', 'OK'), 'cancel': ('إلغاء', 'Cancel'), 'run': ('تشغيل الكشف', 'Run detection'),
    'detect_title': ('الكشف الأوتوماتيكي', 'Automatic detection'),
    'detect_intro': ('البرنامج بيقرأ الصورة وبيرسم الجدران والأعمدة والأبواب والشبابيك والأثاث. الخطوط اللي ما بيفهمها بتنحط بطبقة A-TRACE لحالها.',
                     'Reads the image and draws walls, columns, doors, windows and furniture. Lines it cannot classify go to the separate A-TRACE layer.'),
    'threshold': ('حساسية الحبر', 'Ink threshold'), 'wall_dark': ('أفتح خط مسموح للجدار', 'Lightest wall line'),
    'wall_t': ('سماكة الجدران (م)', 'Wall thickness (m)'), 'min': ('من', 'min'), 'max': ('إلى', 'max'),
    'wall_len': ('أقصر جدار (م)', 'Shortest wall (m)'), 'replace': ('استبدال الرسم الحالي', 'Replace current drawing'),
    'what': ('شو بدك يكتشف؟', 'What to detect'),
    'w_walls': ('جدران', 'Walls'), 'w_columns': ('أعمدة', 'Columns'), 'w_doors': ('أبواب', 'Doors'),
    'w_windows': ('شبابيك', 'Windows'), 'w_furn': ('أثاث (تجريبي)', 'Furniture (experimental)'),
    'working': ('جاري الكشف…', 'Detecting…'),
    'calib_title': ('معايرة المقياس', 'Calibrate scale'),
    'calib_ask': ('المسافة الحقيقية بين النقطتين (متر):', 'Real distance between the two points (m):'),
    'calib_done': ('تم ضبط المقياس: {:.4f} م/بكسل', 'Scale set: {:.4f} m/px'),
    'export_title': ('تصدير DXF', 'Export DXF'), 'inc_scan': ('إرفاق صورة السكان (طبقة A-SCAN مطفية)', 'Attach scan image (A-SCAN, off)'),
    'inc_trace': ('إرفاق الخطوط الخام (طبقة A-TRACE مطفية)', 'Include raw lines (A-TRACE, off)'),
    'exported': ('تم التصدير: {}', 'Exported: {}'),
    'saved': ('تم الحفظ: {}', 'Saved: {}'),
    'text_ask': ('اكتب النص:', 'Enter text:'), 'level_ask': ('قيمة المنسوب (مثلاً +0.45):', 'Level value (e.g. +0.45):'),
    'unsaved': ('في تعديلات ما انحفظت. بدك تحفظ؟', 'There are unsaved changes. Save them?'),
    'no_image': ('افتح صورة أولاً.', 'Open an image first.'),
    'scale_warn': ('المقياس لسا مش معايَر. إذا المخطط عليه أبعاد مكتوبة، البرنامج بيقرأها وبيضبط المقياس لحاله. إذا ما عليه، الأفضل تعاير قبل الكشف. بدك تكمل؟',
                   'Scale is not calibrated. If the drawing has written dimensions they will be read and used. Otherwise calibrate first. Continue?'),
    'coords': ('س {:.2f}  ص {:.2f} م', 'x {:.2f}  y {:.2f} m'),
    'scale_lbl': ('المقياس {:.4f} م/بكسل', 'Scale {:.4f} m/px'),
    'not_calib': (' (غير معايَر)', ' (not calibrated)'),
    'detected': ('تم الكشف: {} جدار، {} عمود، {} باب، {} شباك، {} قطعة أثاث',
                 'Detected: {} walls, {} columns, {} doors, {} windows, {} furniture'),
    'clear_ask': ('اختر الطبقة اللي بدك تمسح كل عناصرها:', 'Choose the layer to clear:'),
    'cat_furn': ('أثاث', 'Furniture'),
    'shortcuts_text': ('''<b>الأدوات</b><br>V تحديد — W جدار — D باب — B باب مزدوج — N شباك — C عمود — S درج — P خط حر — T نص — L منسوب — K معايرة<br><br>
<b>العناصر المحددة</b><br>Del حذف — R تدوير 90° — Q / E تدوير ‎-15° / +15° — M عكس — F قلب فتحة الباب — H تبديل المفصّلة<br><br>
<b>العرض</b><br>عجلة الماوس تكبير — زر الماوس الأوسط أو Space + سحب للتحريك — Ctrl+0 ملاءمة<br><br>
<b>عام</b><br>Ctrl+Z تراجع — Ctrl+Y إعادة — Ctrl+S حفظ — Ctrl+E تصدير DXF — Ctrl+D كشف أوتوماتيكي''',
                       '''<b>Tools</b><br>V select — W wall — D door — B double door — N window — C column — S stair — P polyline — T text — L level — K calibrate<br><br>
<b>Selection</b><br>Del delete — R rotate 90° — Q / E rotate -15° / +15° — M mirror — F flip door swing — H swap hinge<br><br>
<b>View</b><br>Mouse wheel zoom — middle button or Space + drag to pan — Ctrl+0 fit<br><br>
<b>General</b><br>Ctrl+Z undo — Ctrl+Y redo — Ctrl+S save — Ctrl+E export DXF — Ctrl+D auto detect'''),
    'about_text': ('<b>Plan2CAD 1.0</b><br>تحويل صور المخططات المعمارية إلى ملفات DXF بطبقات وبلوكات.<br><br>تطوير: <b>عصام النوافلة</b> — Essam Nawafleh',
                   '<b>Plan2CAD 1.0</b><br>Turns scanned architectural plans into layered DXF files with blocks.<br><br>Developed by <b>Essam Nawafleh</b> — عصام النوافلة'),
    'p_dims': ('قراءة الأبعاد', 'Reading dimensions'),
    'scale_doors': ('ما في أبعاد مكتوبة، فقدّرت المقياس من عرض الأبواب: {:.4f} م/بكسل (للدقة الكاملة عاير المقياس بخط معروف طوله)', 'No dimensions found; scale estimated from door widths: {:.4f} m/px (calibrate with a known length for full accuracy)'),
    'perspective': ('تصحيح المنظور (صور الموبايل)', 'Perspective (phone photos)'),
    'persp_auto': ('تلقائي إذا انلقت حدود الورقة', 'Automatic when the page edge is found'),
    'persp_off': ('بدون', 'Off'),
    'auto_orient': ('تدوير تلقائي حسب اتجاه الكتابة (أبطأ)', 'Auto-orient from text direction (slower)'),
    'denoise_auto': ('إزالة التشويش إذا الصورة مشوّشة', 'Remove noise when the image is noisy'),
    'scale_known': ('بعرف المقياس', 'I know the scale'),
    'compare': ('مقارنة الأصلية بالمعالجة', 'Compare original / processed'),
    'compare_title': ('الصورة الأصلية والمعالجة', 'Original and processed image'),
    'original': ('الأصلية', 'Original'), 'processed': ('بعد المعالجة', 'Processed'),
    'pp_steps': ('خطوات المعالجة', 'Processing steps'),
    'p_nn': ('الشبكة العصبية تقرأ المخطط…', 'Neural network reading the plan…'),
    'p_scale': ('تقدير المقياس من الأبواب…', 'Estimating scale from doors…'),
    'p_validate': ('فحص الهندسة…', 'Checking geometry…'),
    'p_preprocess': ('تجهيز الصورة…', 'Preparing image…'),
    'dims_found': ('قرأت {} رقم، وتأكّدت من {} بُعد على خطوطها. المقياس من الأبعاد: {:.4f} م/بكسل', 'Read {} numbers, confirmed {} dimensions on their lines. Scale from dimensions: {:.4f} m/px'),
    'read_dims': ('اقرأ الأبعاد أولاً (OCR) واضبط المقياس منها', 'Read dimensions first (OCR) and set the scale from them'),
    'p_complete': ('تكميل الجدران من الأبعاد', 'Completing walls from dimensions'),
    'completed': ('ضفت {} جدار من الأبعاد', 'added {} walls from dimensions'),
    'p_fit': ('ضبط الجدران على الأبعاد', 'Fitting walls to dimensions'),
    'fit_dims': ('ضبط الجدران على الأبعاد', 'Fit walls to dimensions'),
    'fit_done': ('ضبطت {} جدار على {} بُعد: متوسط الفرق كان {:.1f} سم وصار {:.1f} سم (أكبر فرق {:.1f} سم)',
                 'Moved {} walls to match {} dimensions: mean error {:.1f} cm → {:.1f} cm (max {:.1f} cm)'),
    'fit_none': ('ما في أبعاد مربوطة بجدران عشان أضبط عليها.', 'No dimensions linked to walls to fit.'),
    'dim_report': ('تقرير الأبعاد', 'Dimension report'),
    'rep_cols': ('البُعد المكتوب|المرسوم|الفرق (سم)', 'Written|Drawn|Diff (cm)'),
    'rep_summary': ('{} بُعد مطابق (أقل من 2 سم)، {} مش مطابق، {} مش مربوط بجدارين', '{} match (< 2 cm), {} off, {} not linked to two walls'),
    'p_chains': ('الشبابيك من سلاسل الأبعاد', 'Windows from dimension chains'),
    'chains_done': ('{} شباك بأبعادها المكتوبة من {} سلسلة', '{} windows at their written sizes from {} chains'),
    'chain_windows': ('حط الشبابيك من سلاسل أبعاد الواجهات', 'Place windows from façade dimension chains'),
    'p_binarize': ('تجهيز', 'Preparing'), 'p_trace': ('تتبّع الخطوط', 'Tracing lines'),
    'p_angles': ('اتجاهات الرسم', 'Drawing directions'), 'p_walls': ('الجدران', 'Walls'),
    'p_columns': ('الأعمدة', 'Columns'), 'p_doors': ('الأبواب', 'Doors'), 'p_windows': ('الشبابيك', 'Windows'),
    'p_furniture': ('الأثاث', 'Furniture'), 'p_done': ('تم', 'Done'),
}

LAYER_NAMES = {
    'A-SCAN': ('صورة السكان', 'Scan image'), 'A-TRACE': ('خطوط خام (غير مصنّفة)', 'Raw lines (unclassified)'),
    'A-WALL': ('جدران', 'Walls'), 'A-COLUMN': ('أعمدة', 'Columns'), 'A-WINDOW': ('شبابيك', 'Windows'),
    'A-DOOR': ('أبواب', 'Doors'), 'A-STAIR': ('أدراج', 'Stairs'), 'A-FURN': ('أثاث', 'Furniture'),
    'A-SANITARY': ('أدوات صحية', 'Sanitary'), 'A-KITCHEN': ('مطابخ', 'Kitchen'),
    'A-COURT': ('كورت وتراس', 'Court & terrace'), 'A-TEXT': ('نصوص', 'Text'), 'A-LEVEL': ('مناسيب', 'Levels'), 'A-DIM': ('أبعاد مقروءة', 'Read dimensions'),
}


def tr(key):
    v = S.get(key)
    if v is None:
        return key
    return v[0] if LANG['cur'] == 'ar' else v[1]


def layer_name(code):
    v = LAYER_NAMES.get(code, (code, code))
    return v[0] if LANG['cur'] == 'ar' else v[1]


def is_rtl():
    return LANG['cur'] == 'ar'
