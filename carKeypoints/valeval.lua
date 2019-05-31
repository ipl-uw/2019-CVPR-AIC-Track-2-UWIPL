------------------------------------------------------------------------------
-- Helpful functions for evaluation
-------------------------------------------------------------------------------

-- Load predictions from hdf5 file
-- predFile: name to the .h5 file containing the predictions
-- doHm: read the 'heatmaps' field of the hdf5 database
-- doInp: read the 'input' field of the hdf5 database
function loadPreds(predFile, doHm, doInp)
    local f = hdf5.open(projectDir .. '/exp/' .. predFile .. '.h5','r')
    local inp,hms
    local idxs = f:read('idxs'):all()
    local preds = f:read('preds'):all()
    if doHm then hms = f:read('heatmaps'):all() end
    if doInp then inp = f:read('input'):all() end
    return idxs, preds, hms, inp
end


-- Calculate the distance between the predicted keypoints and the label (ground-truth keypoints)
-- ??? (Check if function signatures (dims) are correctly understood)
-- Inputs
-- preds: N-by-2 tensor of predicted coordinates
-- label: N-by-2 tensor of predicted coordinates
-- normalize: N-by-1 tensor of normalizing factors
-- Output
-- dists: 2-by-N tensor of computed distances between the predictions and the labels
function calcDists(preds, label, normalize)
    -- Initialize a tensor to hold the distances
    local dists = torch.Tensor(preds:size(2), preds:size(1))
    -- Initialize a 2-by-1 tensor to hold the difference between a label and a prediction
    local diff = torch.Tensor(2)
    -- ???
    -- For each keypoint predicted
    for i = 1,preds:size(1) do
        -- ???
        -- For each dimension of the predicted keypoint (x, y)
        for j = 1,preds:size(2) do
            -- If that keypoint is visible in the image, compute the distance
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
            -- If that keypoint is not visible, let the distance be -1
            else
                dists[j][i] = -1
            end
        end
    end
    -- Return the distance
    return dists
end


-- Recover predictions from a heatmap
-- Input
-- hm: heatmap (a 4-D tensor)
-- Output
-- preds: N-by-2 tensor of predicted keypoint locations obtained from maxima on the heatmap
function getPreds(hm)
    -- ??? (assert the following statement)
    -- I'm assuming the 4 heatmap dimensions are for [num images] x [num kps per image] x [height] x [width]
    
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    -- Reshape the heatmap so that [height] and [width] are flattened out to a single dimension
    -- Get the maxima over the third dimension (comprising of the [height * width] flattened values)
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    -- Allocate memory for a tensor to hold X,Y coordinates of maxima locations
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    -- Obtain the X coordinate of each maxima
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    -- Obtain the Y coordinate of each maxima
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    -- Return the predicted locations
    --print(preds:size())

    return preds
end


-- ???
-- Inputs
-- dists: N-by-2 tensor of distances (between predictions and ground-truth)
-- thr: threshold distance below which a detection is to be considered accurate
-- Output
-- percentage of keypoints that lie within the specified threshold (or -1 if no keypoint is visible)
function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = .5 end
    -- Ignore distances that are -1 (since those keypoints are not visible in the image)
    if torch.ne(dists,-1):sum() > 0 then
        -- For all other keypoints, compute the percentage of keypoints that satisfy the distance threshold
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
end


-- Calculate accuracy according to the PCK (Percentage of Correct Keypoints) metric, but use the 
-- ground-truth heatmap, rather than ground-truth X,Y locations
-- Inputs
-- output: output heatmap (from the hourglass network)
-- label: ground-truth heatmap (??? confirm)
-- thr: threshold distance below which a detection is considered correct
-- idxs: average accuracy across 'idxs' is also returned by this function
-- Output
-- multiple values (each is an accuracy). The first value to be returned is the average accuracy 
-- across 'idxs'. This is followed by accuracies for individual keypoints.
function heatmapAccuracy(output, label, thr, idxs)
    -- Compute predictions from the output heatmap (from the hourglass network)
    local preds = getPreds(output)
    -- Get predictions from the label (ground-truth)
    local gt = getPreds(label)
    -- Calculate the distance between the predictions and the labels
    -- The third argument here is the normalizing factor to be applied at each heatmap location
    -- ??? (find out what the /10 is for)
    local dists = calcDists(preds, gt, torch.ones(preds:size(1))*opt.outputRes/10)

    -- Table to store accuracies
    local acc = {}
    -- Variable to store the average accuracy (over specific keypoints as prescribed by 'idxs')
    local avgAcc = 0.0
    -- Number of indices that are bad (inaccurate)
    local badIdxCount = 0

    -- If average accuracy over 'idxs' is not specified
    if not idxs then
        -- Then compute it over all keypoint indices
        for i = 1,dists:size(1) do
            -- Compute the percentage of keypoints that are correct
            acc[i+1] = distAccuracy(dists[i])
            -- If at least one keypoint is correct, add it to the average accuracy
    	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            -- Otherwise, exclude it
            else badIdxCount = badIdxCount + 1 end
        end
        -- Compute the average accuracy for all keypoint indices
        -- In this evaluation, we consider only those images where at least one keypoint is accurately
        -- predicted.
        acc[1] = avgAcc / (dists:size(1) - badIdxCount)
    -- Compute average accuracy only over specified 'idxs'
    else
        for i = 1,#idxs do
            acc[i+1] = distAccuracy(dists[idxs[i]])
	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (#idxs - badIdxCount)
    end
    -- Return the accuracies
    return unpack(acc)
end


-- ???
-- Calculate basic accuracy
-- Inputs
-- output: output coordinates (??? heatmap or coordinates)
-- label: ground-truth keypoint coordinates
-- thr: threshold
-- Output
-- Percentage of correct keypoints
function basicAccuracy(output, label, thr)
    -- Default threshold of .5
    if not thr then thr = .5 end
    -- Flatten both the output and the label
    output = output:view(output:numel())
    label = label:view(label:numel())
    -- Check if the prediction is within the threshold of the label
    local rounded_output = torch.ceil(output - thr):typeAs(label)
    local eql = torch.eq(label,rounded_output):typeAs(label)
    -- Return PCK
    return eql:sum()/output:numel()
end


-- ???
-- Generate standard PCK plot
function displayPCK(dists, part_idx, label, title, show_key)
    
    if not (type(part_idx) == 'table') then
        part_idx = {part_idx}
    end

    curve_res = 11
    num_curves = #dists
    local t = torch.linspace(0,.5,curve_res)
    local pdj_scores = torch.zeros(num_curves, curve_res)
    local plot_args = {}
    print(title)
    for curve = 1,num_curves do
        for i = 1,curve_res do
            t[i] = (i-1)*.05
            local acc = 0.0
            for j = 1,#part_idx do
                acc = acc + distAccuracy(dists[curve][part_idx[j]], t[i])
            end
            pdj_scores[curve][i] = acc / #part_idx
        end
        plot_args[curve] = {label[curve],t,pdj_scores[curve],'-'}
        print(label[curve],pdj_scores[curve][curve_res])
    end

    require 'gnuplot'
    gnuplot.raw('set title "' .. title .. '"')
    if not show_key then gnuplot.raw('unset key') 
    else gnuplot.raw('set key font ",6" right bottom') end
    gnuplot.raw('set xrange [0:.5]')
    gnuplot.raw('set yrange [0:1]')
    gnuplot.plot(unpack(plot_args))
end
