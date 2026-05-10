import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment'; // ✅ FIXED

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {

  // 🔥 Backend URL from environment
  backendUrl = environment.apiUrl;

  selectedFile: File | null = null;
  result: any;
  loading = false;

  heatmapUrl: string = '';

  sampleImages = [
    'assets/sample1.jpg',
    'assets/sample2.jpg',
    'assets/sample3.jpg'
  ];

  constructor(private http: HttpClient) {}

  // 📌 File select
  onFileSelected(event: any) {
    if (event.target.files.length > 0) {
      this.selectedFile = event.target.files[0];
    }
  }

  // 📌 Use sample image
  useSample(imgPath: string) {
    fetch(imgPath)
      .then(res => res.blob())
      .then(blob => {
        this.selectedFile = new File([blob], 'sample.jpg', { type: blob.type });
      });
  }

  // 📌 Predict
  predict() {
    if (!this.selectedFile) return;

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.loading = true;

    this.http.post(`${this.backendUrl}/predict`, formData)
      .subscribe({
        next: (res) => {
          this.result = res;
          this.heatmapUrl = ''; // clear old heatmap
          this.loading = false;
        },
        error: (err) => {
          console.error(err);
          this.loading = false;
        }
      });
  }

  // 📌 Explain (XAI)
  explain() {
    if (!this.selectedFile) return;

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.loading = true;

    this.http.post<any>(`${this.backendUrl}/explain`, formData)
      .subscribe({
        next: (res) => {
          this.result = res;
          this.generateHeatmap(res.ig_map);
          this.loading = false;
        },
        error: (err) => {
          console.error(err);
          this.loading = false;
        }
      });
  }

  // 📌 Convert IG → heatmap image
//   generateHeatmap(igMap: number[][][]) {
//     const canvas = document.createElement('canvas');
//     const ctx = canvas.getContext('2d')!;

//     const size = 224;
//     canvas.width = size;
//     canvas.height = size;

//     const imageData = ctx.createImageData(size, size);

//     for (let i = 0; i < size; i++) {
//       for (let j = 0; j < size; j++) {
//         const pixel = igMap[i][j];

//         const value = Math.floor(pixel[0] * 255);

//         const index = (i * size + j) * 4;

//         imageData.data[index] = value;     // R
//         imageData.data[index + 1] = 0;
//         imageData.data[index + 2] = 0;
//         imageData.data[index + 3] = 255;
//       }
//     }

//     ctx.putImageData(imageData, 0, 0);
//     this.heatmapUrl = canvas.toDataURL();
//   }
// }


  generateHeatmap(igMap: number[][][]) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;

  const size = 224;
  canvas.width = size;
  canvas.height = size;

  const imageData = ctx.createImageData(size, size);

  // 🔥 Step 1: find min & max
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const val = igMap[i][j][0];
      if (val < min) min = val;
      if (val > max) max = val;
    }
  }

  const range = max - min || 1;

  // 🔥 Step 2: normalize + color map
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const raw = igMap[i][j][0];

      // normalize 0–1
      const norm = (raw - min) / range;

      const value = Math.floor(norm * 255);

      const index = (i * size + j) * 4;

      imageData.data[index] = value;       // R
      imageData.data[index + 1] = 0;       // G
      imageData.data[index + 2] = 255 - value; // B (🔥 makes gradient!)
      imageData.data[index + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  this.heatmapUrl = canvas.toDataURL();
}
}







// import { Component } from '@angular/core';
// import { HttpClient } from '@angular/common/http';

// @Component({
//   selector: 'app-root',
//   templateUrl: './app.component.html',
//   styleUrls: ['./app.component.scss']
// })
// export class AppComponent {

//   backendUrl = 'http://127.0.0.1:8000';

//   selectedFile: File | null = null;
//   result: any;
//   loading = false;

//   sampleImages = [
//     'assets/sample1.jpg',
//     'assets/sample2.jpg',
//     'assets/sample3.jpg'
//   ];

//   constructor(private http: HttpClient) {}

//   // ✅ File upload from input
//   onFileSelected(event: any) {
//     if (event.target.files.length > 0) {
//       this.selectedFile = event.target.files[0];
//     }
//   }

//   // ✅ Use sample image
//   useSample(imgPath: string) {
//     fetch(imgPath)
//       .then(res => res.blob())
//       .then(blob => {
//         this.selectedFile = new File([blob], 'sample.jpg', { type: blob.type });
//       })
//       .catch(err => {
//         console.error("Sample image load error:", err);
//       });
//   }

//   // ✅ Call backend
//   predict() {
//     if (!this.selectedFile) {
//       alert("Please select an image first");
//       return;
//     }

//     const formData = new FormData();
//     formData.append('file', this.selectedFile);

//     this.loading = true;

//     this.http.post(`${this.backendUrl}/predict`, formData)
//       .subscribe({
//         next: (res) => {
//           this.result = res;
//           this.loading = false;
//         },
//         error: (err) => {
//           console.error("API Error:", err);
//           alert("Error from backend");
//           this.loading = false;
//         }
//       });
//   }
// }
