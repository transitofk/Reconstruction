#import openpyxl

"""
def save_book(name, index, name_file, frames, gmsd_results, mse_results, psnr_results, ssim_results):
    book = openpyxl.Workbook()
    name_sheet = book.create_sheet(f'{name}', index)
    if index == 0:
        name_sheet.append(['Frames count'])
    else:
        name_sheet.append(['Cycle count'])
    name_sheet.append(frames.tolist())
    name_sheet.append(['GMSDs'])
    name_sheet.append(gmsd_results)
    name_sheet.append(['MSEs'])
    name_sheet.append(mse_results)
    name_sheet.append(['PSNRs'])
    name_sheet.append(psnr_results)
    name_sheet.append(['SSIMs'])
    name_sheet.append(ssim_results)
    book.save(name_file)"""