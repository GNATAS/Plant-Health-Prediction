// app/static/js/app.js

const API_BASE_URL = '/api';

// เมื่อหน้าเว็บโหลดเสร็จ ให้ดึงข้อมูลแปลงปลูกมาแสดงทันที
document.addEventListener('DOMContentLoaded', () => {
    loadFields();
});

// ==========================================
// 1. ระบบจัดการแปลงปลูก (Field Management)
// ==========================================

// ฟังก์ชันดึงข้อมูลแปลงทั้งหมดมาแสดงที่ List และ Dropdown
async function loadFields() {
    try {
        const response = await fetch(`${API_BASE_URL}/fields/`);
        const fields = await response.json();

        const fieldsList = document.getElementById('fieldsList');
        const fieldSelect = document.getElementById('fieldSelect');

        // เคลียร์ข้อมูลเก่า
        fieldsList.innerHTML = '';
        fieldSelect.innerHTML = '<option value="" disabled selected>-- กรุณาเลือกแปลง --</option>';

        if (fields.length === 0) {
            fieldsList.innerHTML = '<li class="list-group-item text-center text-muted">ยังไม่มีแปลงปลูก</li>';
            return;
        }

        // วนลูปเอาข้อมูลแปลงมาใส่ในหน้าเว็บ
        fields.forEach(field => {
            // ใส่ใน List ด้านซ้าย
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.innerHTML = `<span><strong>${field.name}</strong> <small class="text-muted">(${field.plant_type})</small></span>`;
            fieldsList.appendChild(li);

            // ใส่ใน Dropdown แบบฟอร์มทำนายผล
            const option = document.createElement('option');
            option.value = field.id;
            option.textContent = `ID: ${field.id} - ${field.name} (${field.plant_type})`;
            fieldSelect.appendChild(option);
        });

    } catch (error) {
        console.error('Error loading fields:', error);
    }
}

// จัดการเมื่อกดปุ่ม "เพิ่มแปลงใหม่"
document.getElementById('addFieldForm').addEventListener('submit', async (e) => {
    e.preventDefault(); // ป้องกันหน้าเว็บกระพริบรีเฟรช

    const fieldName = document.getElementById('fieldName').value;
    const plantType = document.getElementById('plantType').value;

    try {
        const response = await fetch(`${API_BASE_URL}/fields/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: fieldName, plant_type: plantType })
        });

        if (response.ok) {
            alert('✅ เพิ่มแปลงปลูกสำเร็จ!');
            document.getElementById('addFieldForm').reset(); // ล้างฟอร์ม
            loadFields(); // โหลดรายชื่อแปลงใหม่
        } else {
            alert('❌ เกิดข้อผิดพลาดในการเพิ่มแปลง');
        }
    } catch (error) {
        console.error('Error adding field:', error);
    }
});

// ==========================================
// 2. ระบบพยากรณ์สุขภาพพืช (Prediction System)
// ==========================================

// ฟังก์ชันสุ่มตัวเลข (สำหรับปุ่มทดสอบ)
function getRandomArbitrary(min, max) {
    return (Math.random() * (max - min) + min).toFixed(2);
}

// จัดการเมื่อกดปุ่ม "สุ่มข้อมูลทดสอบ"
document.getElementById('btnFillDummy').addEventListener('click', () => {
    // ใส่ค่าสุ่มที่อยู่ในเกณฑ์ที่ Model ของคุณน่าจะรู้จัก
    document.getElementById('soilMoisture').value = getRandomArbitrary(10, 40);
    document.getElementById('ambientTemp').value = getRandomArbitrary(20, 35);
    document.getElementById('soilTemp').value = getRandomArbitrary(18, 30);
    document.getElementById('humidity').value = getRandomArbitrary(40, 80);
    document.getElementById('lightIntensity').value = getRandomArbitrary(200, 800);
    document.getElementById('soilPh').value = getRandomArbitrary(5.5, 7.5);
    document.getElementById('nitrogenLevel').value = getRandomArbitrary(10, 50);
    document.getElementById('phosphorusLevel').value = getRandomArbitrary(10, 50);
    document.getElementById('potassiumLevel').value = getRandomArbitrary(10, 50);
    document.getElementById('chlorophyllContent').value = getRandomArbitrary(20, 60);
    document.getElementById('electroSignal').value = getRandomArbitrary(0.5, 2.0);
});

// จัดการเมื่อกดปุ่ม "ทำนายผล"
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fieldId = document.getElementById('fieldSelect').value;
    if (!fieldId) {
        alert("กรุณาเลือกแปลงปลูกก่อนทำนายผล!");
        return;
    }

    // รวบรวมข้อมูลจากฟอร์มให้ตรงกับ Schema ใน Backend
    const payload = {
        field_id: parseInt(fieldId),
        soil_moisture: parseFloat(document.getElementById('soilMoisture').value),
        ambient_temperature: parseFloat(document.getElementById('ambientTemp').value),
        soil_temperature: parseFloat(document.getElementById('soilTemp').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        light_intensity: parseFloat(document.getElementById('lightIntensity').value),
        soil_ph: parseFloat(document.getElementById('soilPh').value),
        nitrogen_level: parseFloat(document.getElementById('nitrogenLevel').value),
        phosphorus_level: parseFloat(document.getElementById('phosphorusLevel').value),
        potassium_level: parseFloat(document.getElementById('potassiumLevel').value),
        chlorophyll_content: parseFloat(document.getElementById('chlorophyllContent').value),
        electrochemical_signal: parseFloat(document.getElementById('electroSignal').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/predictions/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const result = await response.json();
            showResult(result);
            loadHistory(fieldId); // โหลดประวัติใหม่ทันที
        } else {
            const err = await response.json();
            alert(`❌ Error: ${err.detail}`);
        }
    } catch (error) {
        console.error('Prediction Error:', error);
        alert('❌ เกิดข้อผิดพลาด ไม่สามารถเชื่อมต่อกับระบบทำนายผลได้');
    }
});

// แสดงผลลัพธ์ในหน้าจอ
function showResult(result) {
    const resultArea = document.getElementById('resultArea');
    const statusBadge = document.getElementById('predictResultStatus');
    const probaDiv = document.getElementById('predictProbabilities');

    // โชว์พื้นที่ผลลัพธ์
    resultArea.classList.remove('d-none');

    // ตั้งค่าสีป้าย (Badge) ตาม Status
    statusBadge.textContent = result.predicted_status;
    statusBadge.className = 'badge ';
    if (result.predicted_code === 0) statusBadge.classList.add('bg-success'); // Healthy
    else if (result.predicted_code === 1) statusBadge.classList.add('bg-warning', 'text-dark'); // Moderate
    else statusBadge.classList.add('bg-danger'); // High Stress

    // แสดงค่าความน่าจะเป็น (ถ้ามี)
    probaDiv.innerHTML = '';
    if (result.probabilities) {
        const probs = JSON.parse(result.probabilities);
        for (const [cls, prob] of Object.entries(probs)) {
            const percent = (prob * 100).toFixed(2);
            probaDiv.innerHTML += `<div>${cls}: <strong>${percent}%</strong></div>`;
        }
    }
}

// ==========================================
// 3. ระบบประวัติการพยากรณ์ (History)
// ==========================================

// เมื่อเปลี่ยนแปลงใน Dropdown ให้โหลดประวัติของแปลงนั้นมาแสดง
document.getElementById('fieldSelect').addEventListener('change', (e) => {
    if (e.target.value) {
        loadHistory(e.target.value);
    }
});

// เมื่อกดปุ่มรีเฟรชประวัติ
document.getElementById('btnRefreshHistory').addEventListener('click', () => {
    const fieldId = document.getElementById('fieldSelect').value;
    if (fieldId) {
        loadHistory(fieldId);
    } else {
        alert("กรุณาเลือกแปลงปลูกในส่วนทำนายผลก่อน เพื่อดูประวัติ");
    }
});

// ฟังก์ชันโหลดประวัติจาก API
async function loadHistory(fieldId) {
    const tbody = document.getElementById('historyTableBody');
    if (!tbody) return; // กันพลาดถ้าหาตารางไม่เจอ

    tbody.innerHTML = '<tr><td colspan="5" class="text-center">กำลังดึงประวัติ...</td></tr>';

    try {
        const response = await fetch(`${API_BASE_URL}/fields/${fieldId}`);
        const data = await response.json();
        
        // ข้อมูลที่คุณส่งมาอยู่ใน data.predictions
        const history = data.predictions; 

        if (!history || history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-4">ยังไม่มีประวัติการทำนาย</td></tr>';
            return;
        }

        tbody.innerHTML = ''; // ล้างแถว "กำลังโหลด" ออก

        // เรียงใหม่ไปเก่า
        history.reverse().forEach(row => {
            const dateStr = new Date(row.created_at).toLocaleString('th-TH');
            
            // กำหนดสี Badge
            let colorClass = 'bg-secondary';
            if (row.predicted_code === 0) colorClass = 'bg-success';
            else if (row.predicted_code === 1) colorClass = 'bg-warning text-dark';
            else if (row.predicted_code === 2) colorClass = 'bg-danger';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="small">${dateStr}</td>
                <td><strong>${data.name}</strong></td>
                <td colspan="2" class="text-center small text-muted">บันทึกสำเร็จ (ID: ${row.id})</td>
                <td><span class="badge ${colorClass}">${row.predicted_status}</span></td>
            `;
            tbody.appendChild(tr);
        });

    } catch (error) {
        console.error('Error:', error);
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">โหลดประวัติไม่สำเร็จ</td></tr>';
    }
}