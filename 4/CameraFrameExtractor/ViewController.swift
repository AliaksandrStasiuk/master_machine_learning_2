//
//  ViewController.swift
//  CameraFrameExtractor
//
//  Created by Anand Nigam on 11/10/18.
//  Copyright © 2018 Anand Nigam. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    @IBOutlet weak var cameraPreview: UIView!
    
    let targetImageSize = CGSize(width: 208, height: 352)
    var videoDataOutput: AVCaptureVideoDataOutput!
    var videoDataOutputQueue: DispatchQueue!
    var previewLayer:AVCaptureVideoPreviewLayer!
    var captureDevice : AVCaptureDevice!
    let session = AVCaptureSession()
    let context = CIContext()
    
    @IBOutlet weak var textDigit: UITextField!
    @IBOutlet weak var labelDigit: UILabel!
    var i = Int(0)
    
    var seconds_before = Int(0)
    
    var model = try! VNCoreMLModel( for: model_digit().model);
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        view.addSubview(cameraPreview)
        self.setUpAVCapture()
        let date = Date()
        let calendar = Calendar.current
        let minutes = calendar.component(.minute, from: date)
        self.seconds_before = calendar.component(.second, from: date) + minutes*60
        
        if #available(iOS 13.0, *) {
            print("Should be this!")
            self.model = try! VNCoreMLModel( for: model_digit().model);
        }
        
    }
    
    override func viewWillDisappear(_ animated: Bool) {
//        super.viewWillDisappear(animated)
        stopCamera()
    }

    // To add the layer of your preview
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.cameraPreview.layer.bounds
    }
    
    // To set the camera and its position to capture
    func setUpAVCapture() {
        session.sessionPreset = AVCaptureSession.Preset.hd1280x720
        guard var device = AVCaptureDevice
            .default(AVCaptureDevice.DeviceType.builtInWideAngleCamera,
            for: .video,
            position: AVCaptureDevice.Position.back
        ) else {
                        return
        }
//        var format:AVCaptureDevice.Format?
//        for vFormat in device.formats {
////            print(vFormat.highResolutionStillImageDimensions)
//            if vFormat.highResolutionStillImageDimensions.height == 720 {
//                format = vFormat
//                break
//            }
//        }
//        print(format!)
//        try! device.lockForConfiguration()
//        device.activeFormat = format!
//        device.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: 240)
//        device.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: 240)
//        device.unlockForConfiguration()
        captureDevice = device
        beginSession()
    }
    
    // Function to setup the beginning of the capture session
    func beginSession(){
        var deviceInput: AVCaptureDeviceInput!
        
        do {
            deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            guard deviceInput != nil else {
                print("error: cant get deviceInput")
                return
            }
            
            if self.session.canAddInput(deviceInput){
                self.session.addInput(deviceInput)
            }
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.alwaysDiscardsLateVideoFrames=true
            videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
            videoDataOutput.setSampleBufferDelegate(self, queue:self.videoDataOutputQueue)
            
            if session.canAddOutput(self.videoDataOutput){
                session.addOutput(self.videoDataOutput)
            }
            
            videoDataOutput.connection(with: .video)?.isEnabled = true
            
            previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
//            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
            
            let rootLayer :CALayer = self.cameraPreview.layer
            rootLayer.masksToBounds=true
            
            rootLayer.addSublayer(self.previewLayer)
            session.startRunning()
        } catch let error as NSError {
            deviceInput = nil
            print("error: \(error.localizedDescription)")
        }
    }
    
    // Function to capture the frames again and again
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // do stuff here
//        print("Got a frame")
        
        DispatchQueue.main.async { [unowned self] in
            guard let uiImage = self.imageFromSampleBuffer(sampleBuffer: sampleBuffer) else { return }
            self.detectPhoto(image: uiImage)
        }
        
    }
    
    // Function to process the buffer and return UIImage to be used
    func imageFromSampleBuffer(sampleBuffer : CMSampleBuffer) -> UIImage? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        
        
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        
        return UIImage(cgImage: cgImage)
    }
    
    // To stop the session 
    func stopCamera(){
        session.stopRunning()
    }
    
    func convertToGrayScale(image: UIImage) -> UIImage {

        // Create image rectangle with current image width/height
        let imageRect:CGRect = CGRect(x:0, y:0, width:image.size.width, height: image.size.height)

        // Grayscale color space
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let width = image.size.width
        let height = image.size.height

        // Create bitmap content with current image size and grayscale colorspace
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        // Draw image into current context, with specified rectangle
        // using previously defined context (with grayscale colorspace)
        let context = CGContext(data: nil, width: Int(width), height: Int(height), bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
        context?.draw(image.cgImage!, in: imageRect)
        let imageRef = context!.makeImage()

        // Create a new UIImage object
        let newImage = UIImage(cgImage: imageRef!)

        return newImage
    }
    
    func getDigit(arr:MLMultiArray) -> Int {
        var max_conf:Double = 0
        var index:Int = 0
//        print(arr.shape[0].intValue)
        for ind in 0...arr.shape[0].intValue{
            if arr[ind].doubleValue < 1.1 {
                if arr[ind].doubleValue > max_conf{
                    max_conf = arr[ind].doubleValue
                    index = ind
                }
            }
        }
//        print(index)
        if index == 9{
            index = 0
        }
        if index == 10 {
            index = 0
        }
        if max_conf < 0.999 {
            index = 0
        }
//        print(max_conf)
        return index
    }
    
    func updateLabel(string: String){
        labelDigit.text = string
    }

  
    func detectPhoto(image: UIImage){
        
//        var flippedImage:UIImage = image.withHorizontallyFlippedOrientation()
        
//        flippedImage.imageOrientation = UIImage.Orientation.right
        
//        let image_gray = self.convertToGrayScale(image: image)
        
        guard var ciImage = CIImage(image: image) else {
            fatalError("Error")
        }
        
        let request = VNCoreMLRequest(model: self.model) {
            (vnRequest, error) in vnRequest.results
//            print(type(vnRequest))
                guard let results = vnRequest.results as?
                    [VNCoreMLFeatureValueObservation] else {
                        fatalError("unexpected result")
                }
            
            let pos3: MLMultiArray? = results[4].featureValue.multiArrayValue
            let pos5: MLMultiArray? = results[0].featureValue.multiArrayValue
            let pos2: MLMultiArray? = results[1].featureValue.multiArrayValue
            let pos4: MLMultiArray? = results[2].featureValue.multiArrayValue
            let pos1: MLMultiArray? = results[3].featureValue.multiArrayValue
            
            let digit3:Int = self.getDigit(arr:pos3!)
            let digit5:Int = self.getDigit(arr:pos5!)
            let digit2:Int = self.getDigit(arr:pos2!)
            let digit4:Int = self.getDigit(arr:pos4!)
            let digit1:Int = self.getDigit(arr:pos1!)
            
            let strin: String = String(digit1) + String(digit2) + String(digit3) +
            String(digit4) + String(digit5)
            
            print(strin)
            DispatchQueue.main.async {
                self.updateLabel(string: strin)
            }
            
        }
        
//        request.imageCropAndScaleOption = .centerCrop
            
        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: .right, options: [:])
        DispatchQueue.global(qos:
            DispatchQoS.QoSClass.userInteractive).async {
                do{
                    try handler.perform([request])
                } catch {
                    print(error)
                }
        }
    }
}
