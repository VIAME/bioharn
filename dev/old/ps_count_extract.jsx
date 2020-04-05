#target photoshop


// Proof of concept write to a file.
// This file should be saved based on the name of the image, so we can assocate
// the annotations to the original image. Ideally we also save the image path 
// in the annotation file that we export. 
var saveFile = new File("C:\\Users\\steve.jordan\\Desktop\\MyPsScript\\foo.txt");
saveFile.encoding = "UTF8";
saveFile.open("w");


if (app.documents.length > 0) {

    var ref = new ActionReference();
    ref.putEnumerated( charIDToTypeID("Dcmn"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt") );
    var docDesc = executeActionGet(ref);

    if (docDesc.hasKey(stringIDToTypeID("countClass")) == true) {

        // set to 72dpi and pixels;
        var originalRulerUnits = preferences.rulerUnits;
        preferences.rulerUnits = Units.PIXELS;
        var myDocument = app.activeDocument;
        var originalResolution = myDocument.resolution;
        myDocument.resizeImage(undefined, undefined, 72, ResampleMethod.NONE);
        // get coordinates;
        var counter =docDesc.getList(stringIDToTypeID("countClass"));
        var thePoints = new Array;
        for (var c = 0; c < counter.count; c++) {
            var thisOne = counter.getObjectValue(c);

            // Proof of concept that we can write out x/y positions of count annotations
            // TODO: we also need to write out the object category
            // TODO: write this out to a nice MS-COCO-like json file.
            saveFile.write("\"x\": " + thisOne.getUnitDoubleValue(stringIDToTypeID("x")) + ", ")
            saveFile.write("\"y\": " + thisOne.getUnitDoubleValue(stringIDToTypeID("y")) + ", ")
            saveFile.write("\n")

            thePoints.push([thisOne.getUnitDoubleValue(stringIDToTypeID("x")), thisOne.getUnitDoubleValue(stringIDToTypeID("y"))])
        };

    }

    saveFile.write("hello word")
    saveFile.close();
}

