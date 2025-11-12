using Plots
using Random
using DelimitedFiles
using Interpolations
using Statistics
using LaTeXStrings


function create_interpolated_function(filename::String)
    if !isfile(filename)
        throw(ArgumentError("File not found: $filename"))
    end

    local data_matrix
    try
        # Read the file as a matrix of numbers.
        # `readdlm` will try to parse numbers, and if it fails, it might return strings
        # or throw an error depending on the content.
        # Changed delimiter from ' ' to '\t' to handle tab-separated values.
        data_matrix = readdlm(filename, '\t', Float64, '\n', header=false)
    catch e
        throw(ErrorException("Error reading or parsing file '$filename': $e. Ensure it contains only numeric data and is tab-separated."))
    end

    if size(data_matrix, 1) < 2
        throw(ArgumentError("File '$filename' must contain at least two lines of data (x and y values)."))
    end

    # Extract x and y values from the first two rows
    x_values = vec(data_matrix[1, :]) # Ensure it's a 1D vector
    y_values = vec(data_matrix[2, :]) # Ensure it's a 1D vector

    if length(x_values) != length(y_values)
        throw(ArgumentError("The number of x values ($(length(x_values))) does not match the number of y values ($(length(y_values))) in file '$filename'."))
    end

    if isempty(x_values) || isempty(y_values)
        throw(ArgumentError("File '$filename' contains no data points for interpolation."))
    end

    # Create a linear interpolation object
    # `LinearInterpolation` requires sorted x values. If your x values are not guaranteed
    # to be sorted, you might need to sort them along with their corresponding y values.
    # For simplicity, this example assumes x_values are already sorted or will be sorted.
    # If not sorted, uncomment the following block:
    # p = sortperm(x_values)
    # x_values = x_values[p]
    # y_values = y_values[p]

    # By default, `LinearInterpolation` will throw a `BoundsError` if queried outside the range.
    itp = LinearInterpolation(x_values, y_values)

    return itp, x_values, y_values # Return the interpolation, and the original x and y values
end




function sample_from_interpolated_pdf(itp_func, x_grid::Vector{Float64}, num_samples::Int)
    if isempty(x_grid)
        throw(ArgumentError("x_grid cannot be empty."))
    end
    if !issorted(x_grid)
        throw(ArgumentError("x_grid must be sorted to calculate CDF correctly."))
    end

    # 1. Evaluate the interpolated function at the grid points
    y_vals_at_grid = itp_func.(x_grid)

    # 2. Ensure y-values are non-negative for probability interpretation
    if any(y < 0 for y in y_vals_at_grid)
        #@warn "Interpolated function has negative values, treating them as zero for sampling."
        y_vals_at_grid = max.(0.0, y_vals_at_grid)
    end

    # 3. Calculate the approximate area under the curve for normalization (using trapezoidal rule)
    area = 0.0
    for i in 1:(length(x_grid)-1)
        area += 0.5 * (y_vals_at_grid[i] + y_vals_at_grid[i+1]) * (x_grid[i+1] - x_grid[i])
    end

    if area <= 0
        throw(ArgumentError("The area under the interpolated function is zero or negative. Cannot sample from this distribution."))
    end

    # 4. Calculate the Cumulative Distribution Function (CDF) values
    cdf_values = zeros(length(x_grid))
    cumulative_area = 0.0
    for i in 1:(length(x_grid)-1)
        segment_area = 0.5 * (y_vals_at_grid[i] + y_vals_at_grid[i+1]) * (x_grid[i+1] - x_grid[i])
        cumulative_area += segment_area
        cdf_values[i+1] = cumulative_area / area # Normalize to [0, 1]
    end
    # Ensure the last CDF value is exactly 1.0 due to potential floating point inaccuracies
    cdf_values[end] = 1.0

    # 5. Create an interpolated function for the CDF
    # Use Flat extrapolation to handle values exactly at 0 or 1, or slightly outside due to precision
    cdf_itp = LinearInterpolation(x_grid, cdf_values, extrapolation_bc=Flat())

    # 6. Create the inverse CDF function (for inverse transform sampling)
    # The cdf_values are already sorted if x_grid is sorted and y_vals_at_grid are non-negative.
    inverse_cdf_itp = LinearInterpolation(cdf_values, x_grid, extrapolation_bc=Flat())

    # 7. Generate samples using inverse transform sampling
    samples = zeros(num_samples)
    for i in 1:num_samples
        u = rand() # Generate a uniform random number between 0 and 1
        samples[i] = inverse_cdf_itp(u)
    end

    return samples
end

function get_parton_number(val_u,val_u_x_data, val_d, val_d_x_data,sea_g, sea_g_x_data, sea_d, sea_d_x_data, sea_db, sea_db_x_data,sea_u, sea_u_x_data, sea_ub, sea_ub_x_data,sea_s, sea_s_x_data, sea_sb, sea_sb_x_data)
    upValMom = sample_from_interpolated_pdf(val_u, val_u_x_data,2)
    downValMom = sample_from_interpolated_pdf(val_d, val_d_x_data,1)
    x_sum=sum(upValMom) + sum(downValMom)
    @show x_sum
    nParton = 3
    if x_sum > 1.0
        return 3
    end
    while x_sum < 1.0
        x_val,n = sample_sea_distr(sea_g, sea_g_x_data, sea_d, sea_d_x_data, sea_db, sea_db_x_data,sea_u, sea_u_x_data, sea_ub, sea_ub_x_data,sea_s, sea_s_x_data, sea_sb, sea_sb_x_data)
        x_sum += x_val
        nParton += n
    end
    return nParton 
end


function QPartonNum(Q,num)


sea_g, sea_g_x_data, sea_g_y_data = create_interpolated_function("./Sampling_PDF/resG_Q"*string(Q)*".txt")
sea_d, sea_d_x_data, sea_d_y_data = create_interpolated_function("./Sampling_PDF/resD_Q"*string(Q)*".txt")
sea_db, sea_db_x_data, sea_db_y_data = create_interpolated_function("./Sampling_PDF/resDB_Q"*string(Q)*".txt")
sea_u, sea_u_x_data, sea_u_y_data = create_interpolated_function("./Sampling_PDF/resU_Q"*string(Q)*".txt")
sea_ub, sea_ub_x_data, sea_ub_y_data = create_interpolated_function("./Sampling_PDF/resUB_Q"*string(Q)*".txt")
sea_s, sea_s_x_data, sea_s_y_data = create_interpolated_function("./Sampling_PDF/resS_Q"*string(Q)*".txt")
sea_sb, sea_sb_x_data, sea_sb_y_data = create_interpolated_function("./Sampling_PDF/resSB_Q"*string(Q)*".txt")

val_d, val_d_x_data, val_d_y_data = create_interpolated_function("./Sampling_PDF/resVD_Q"*string(Q)*".txt")
val_u, val_u_x_data, val_u_y_data = create_interpolated_function("./Sampling_PDF/resVU_Q"*string(Q)*".txt")

histList=map(x->get_parton_number(val_u,val_u_x_data, val_d, val_d_x_data,sea_g, sea_g_x_data, sea_d, sea_d_x_data, sea_db, sea_db_x_data,sea_u, sea_u_x_data, sea_ub, sea_ub_x_data,sea_s, sea_s_x_data, sea_sb, sea_sb_x_data),1:num)

return histList

end

function sample_sea_distr(sea_g, sea_g_x_data, sea_d, sea_d_x_data, sea_db, sea_db_x_data,
                          sea_u, sea_u_x_data, sea_ub, sea_ub_x_data,
                          sea_s, sea_s_x_data, sea_sb, sea_sb_x_data)
    index=rand(0:3)
    if index == 0
        return (sum(sample_from_interpolated_pdf(sea_g, sea_g_x_data, 1)),1)
    elseif index == 1
        return (sum(sample_from_interpolated_pdf(sea_d, sea_d_x_data, 1)) + sum(sample_from_interpolated_pdf(sea_db, sea_db_x_data, 1)),2)
    elseif index == 2
        return (sum(sample_from_interpolated_pdf(sea_u, sea_u_x_data, 1)) + sum(sample_from_interpolated_pdf(sea_ub, sea_ub_x_data, 1)),2)
    elseif index == 3
        return (sum(sample_from_interpolated_pdf(sea_s, sea_s_x_data, 1)) + sum(sample_from_interpolated_pdf(sea_sb, sea_sb_x_data, 1)),2)
    end
end

numEv=500


hist1=QPartonNum(1,numEv)
#hist2=QPartonNum(2,numEv)
#hist3=QPartonNum(3,numEv)
#hist4=QPartonNum(4,numEv)
hist5=QPartonNum(5,numEv)
#hist6=QPartonNum(6,numEv)
#hist7=QPartonNum(7,numEv)
#hist8=QPartonNum(8,numEv)
#hist9=QPartonNum(9,numEv)
hist10=QPartonNum(10,numEv)


mean(hist1)
mean(hist5)
mean(hist10)




meanList=[mean(hist1),mean(hist2),mean(hist3),mean(hist4),mean(hist5),mean(hist6),mean(hist7),mean(hist8),mean(hist9),mean(hist10)]
stdList=[std(hist1),std(hist2),std(hist3),std(hist4),std(hist5),std(hist6),std(hist7),std(hist8),std(hist9),std(hist10)]
Qlist=1:1:10

writedlm("mean_std.dat", hcat(Qlist, meanList, stdList))

avePlot=plot(Qlist,meanList, yaxis=L"\langle m \rangle(Q^2)",label="",xaxis=L"Q^2 \; [GeV^2]")
stdPlot=plot(Qlist,stdList, yaxis=L"\sigma(m)(Q^2)",label="",xaxis=L"Q^2 \; [GeV^2]")

savefig(avePlot,"aveMPlot.pdf")
savefig(stdPlot,"stdMPlot.pdf")

2*pi/(2-log(2))
log(2)

vec1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
vec2 = [46.8285, 50.2984, 52.8984, 53.8655, 53.6298, 53.057, 53.0973, 52.9149, 52.3412, 51.8001]
vec3 = [31.079007033974264, 28.22874498882946, 27.963899502886587, 27.476726639115512, 26.502105352571743, 26.08151405761471, 25.313223987490492, 25.20045960506337, 24.526722948120433, 23.983307943150578]

plot(vec2)
plot(vec3)

hcat(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, hist10)